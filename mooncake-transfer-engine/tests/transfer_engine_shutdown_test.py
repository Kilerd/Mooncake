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
"""TransferEngine.shutdown() and a new engine in the same process.

CPU mode (default, no GPU): a stand-in on a unix socket plays
mooncake_nvlink_proxy (HELLO, PING, REGISTER, RELEASE); the engines use TCP
(MC_FORCE_TCP=1) and host memory.

  - loopback writes work; shutdown() returns 0, is idempotent, and every
    later call raises RuntimeError (initialize() too);
  - once shutdown() returned, the process has the sockets, epoll / eventfd
    descriptors, pipes and threads it had before the engine (what a
    checkpoint would have to carry), and the daemon saw every connection of
    the engine close;
  - a new TransferEngine in the same process initializes and transfers, and
    says HELLO with a new client id and a new process token;
  - a batch left unfreed is canceled, waited for and freed by shutdown();
  - a thread still calling the engine gets RuntimeError, nothing crashes.

GPU mode (--gpu): needs a running mooncake_nvlink_proxy that sees the GPUs,
MC_NVLINK_PROXY_SOCKET pointing at it and MC_NODE_ID set. Two engines of
this process register cudaMalloc memory (on --dev0 / --dev1), write through
the daemon and are shut down: the daemon then holds no IPC mapping of their
memory (--stats), and the process no socket the engines opened (CUDA's own
@cuda-uvmfd-... listener aside). Two new engines in the same process do it
again, through the daemon again. Each round checks every byte. Round 1
registers each buffer with its GPU current, as an engine process does;
round 2 registers both with --dev0 current (the other GPU's buffer must
still be registered with the daemon). With --vmm the memory is two virtual
memory management pools (cuMemCreate, exportable as POSIX file
descriptors, MC_NVLINK_PROXY_VMM=1), whose physical memory is released
after each shutdown and created again at the same addresses -- the flow of
an engine that releases its KV cache before a checkpoint. --require-vmm
(implies --vmm) sets MC_NVLINK_PROXY_REQUIRE_VMM=1 instead and checks that
a cudaMalloc block is refused.

Usage:
  PYTHONPATH=<dir of the mooncake package> transfer_engine_shutdown_test.py
  transfer_engine_shutdown_test.py --gpu [--vmm | --require-vmm]
                                   [--dev0 0] [--dev1 1]
Linux only (/proc/self).
"""

import argparse
import ctypes
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

try:
    from mooncake.engine import TransferEngine
except ImportError:  # a build tree: the module next to the others
    from engine import TransferEngine

MAGIC, VERSION = 0x504E434D, 1
HELLO, REGISTER, PING, RELEASE = 1, 2, 5, 7
OK = 0
MiB = 1 << 20

FAILURES = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        FAILURES.append(what)
    return cond


def resources():
    """Sockets, anon inodes (epoll, eventfd), pipes and threads."""
    out = {"socket": 0, "anon_inode": 0, "pipe": 0}
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        for kind in out:
            if target.startswith(kind + ":"):
                out[kind] += 1
    out["threads"] = len(os.listdir("/proc/self/task"))
    return out


def socket_inodes():
    """Inode -> descriptor of every socket of this process."""
    out = {}
    for fd in os.listdir("/proc/self/fd"):
        try:  # the descriptor listdir itself used is closed by now
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith("socket:["):
            out[target[8:-1]] = fd
    return out


def unix_socket_paths():
    """Inode -> path of the unix sockets (abstract ones start with @)."""
    paths = {}
    try:
        with open("/proc/net/unix") as f:
            next(f)
            for line in f:
                cols = line.split()
                if len(cols) >= 8:
                    paths[cols[6]] = cols[7]
    except OSError:
        pass
    return paths


def sockets_opened_since(before):
    """Sockets of this process that |before| (socket_inodes()) did not
    have, except the driver's listener @cuda-uvmfd-<ns>-<pid>@ (opened at
    the first IPC or virtual memory management export, it lives as long as
    the process): one line each."""
    paths = unix_socket_paths()
    left = []
    for inode, fd in sorted(socket_inodes().items()):
        path = paths.get(inode, "")
        if inode in before or path.startswith("@cuda-uvmfd-"):
            continue
        left.append(f"fd {fd} inode {inode} {path or '(not a named unix socket)'}")
    return left


def resources_settled(want, timeout=2.0):
    """A joined thread is still listed for a moment (pthread_join returns
    when the kernel clears its tid, before the task is gone): waits until
    the process holds |want|, else returns the last reading."""
    end = time.time() + timeout
    r = resources()
    while r != want and time.time() < end:
        time.sleep(0.005)
        r = resources()
    return r


def resources_stable():
    """Two equal readings 50 ms apart."""
    r = resources()
    for _ in range(40):
        time.sleep(0.05)
        nxt = resources()
        if nxt == r:
            break
        r = nxt
    return r


def wait_until(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


# ------------------------------------------------------------- stand-in
class FakeDaemon:
    """Answers like mooncake_nvlink_proxy; records HELLO and RELEASE."""

    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            os.unlink(path)
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(path)
        self.srv.listen(16)
        self.lock = threading.Lock()
        self.hellos = []  # (client id, control, token)
        self.releases = []
        self.open = 0
        self.threads = []
        self.stopping = False
        self.acceptor = threading.Thread(target=self._accept, daemon=True)
        self.acceptor.start()

    def _accept(self):
        while not self.stopping:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            th = threading.Thread(target=self._serve, args=(conn,), daemon=True)
            with self.lock:
                self.open += 1
                self.threads.append(th)
            th.start()

    @staticmethod
    def _read(conn, n):
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _serve(self, conn):
        client = 0
        try:
            while True:
                h = self._read(conn, 24)
                if h is None:
                    break
                magic, ver, typ, _, plen, seq = struct.unpack("<IHHIIQ", h)
                payload = self._read(conn, plen) if plen else b""
                if payload is None:
                    break
                body = b""
                if typ == HELLO:
                    client, _, flags = struct.unpack_from("<QII", payload)
                    token = struct.unpack_from("<Q", payload, 16 + 56)[0]
                    with self.lock:
                        self.hellos.append((client, bool(flags & 1), token))
                    body = struct.pack("<QII", 0x5EED, 1, 0)
                elif typ == PING:
                    body = struct.pack("<QQ", 0x5EED, 1)
                elif typ == RELEASE:
                    with self.lock:
                        self.releases.append(client)
                    body = struct.pack("<QQ", 0, 0)
                conn.sendall(
                    struct.pack("<IHHIIQ", MAGIC, VERSION, typ, OK, len(body), seq)
                    + body
                )
        except OSError:
            pass
        finally:
            conn.close()
            with self.lock:
                self.open -= 1

    def idle(self):
        """No engine connection left, and their threads have exited."""
        with self.lock:
            if self.open:
                return False
            done, self.threads = self.threads, []
        for th in done:
            th.join()
        return True

    def control_clients(self):
        with self.lock:
            return [(c, t) for c, ctl, t in self.hellos if ctl]

    def stop(self):
        self.stopping = True
        self.srv.close()
        os.unlink(self.path)


# ------------------------------------------------------------- CPU mode
class HostBuffer:
    def __init__(self, n, seed=None):
        self.n = n
        self.buf = ctypes.create_string_buffer(n)
        if seed is not None:
            ctypes.memmove(
                self.buf,
                bytes((i * seed + 7) & 0xFF for i in range(256)) * (n // 256),
                n,
            )

    @property
    def addr(self):
        return ctypes.addressof(self.buf)

    def data(self):
        return self.buf.raw


def tcp_engine():
    e = TransferEngine()
    rc = e.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "")
    assert rc == 0, rc
    return e


def loopback_write(engine, n=4 * MiB, seed=3):
    src, dst = HostBuffer(n, seed), HostBuffer(n)
    assert engine.register_memory(src.addr, n) == 0
    assert engine.register_memory(dst.addr, n) == 0
    session = f"127.0.0.1:{engine.get_rpc_port()}"
    rc = engine.batch_transfer_sync_write(session, [src.addr], [dst.addr], [n])
    return rc == 0 and src.data() == dst.data(), (src, dst, session)


def cpu_mode():
    tmp = tempfile.mkdtemp(prefix="mc-shutdown-")
    daemon = FakeDaemon(os.path.join(tmp, "proxy.sock"))
    os.environ["MC_FORCE_TCP"] = "1"
    os.environ["MC_NVLINK_PROXY_SOCKET"] = daemon.path
    try:
        print("== warm-up: what lives as long as the process")
        warm = tcp_engine()
        ok, _ = loopback_write(warm)
        check(ok and warm.shutdown() == 0, "first engine: write, shutdown")
        warm = None  # destroyed after shutdown()
        check(wait_until(daemon.idle), "the daemon saw its connections close")
        before = resources_stable()

        print("== shutdown leaves no connection, listener or thread")
        a = tcp_engine()
        ok, _ = loopback_write(a)
        check(ok, "loopback write")
        during = resources()
        check(
            during["socket"] > before["socket"]
            and during["threads"] > before["threads"],
            f"engine running: {during}",
        )
        t0 = time.monotonic()
        rc = a.shutdown()
        took = time.monotonic() - t0
        check(rc == 0, f"shutdown() == 0 ({took * 1000:.0f} ms)")
        check(wait_until(daemon.idle, 2), "the daemon saw every connection close")
        after = resources_settled(before)
        check(after == before, f"back to {before}: {after}")
        check(a.shutdown() == 0, "shutdown() again == 0")
        for name, call in (
            ("get_rpc_port", lambda: a.get_rpc_port()),
            ("register_memory", lambda: a.register_memory(1 << 30, 4096)),
            (
                "batch_transfer_sync_write",
                lambda: a.batch_transfer_sync_write("127.0.0.1:1", [1], [1], [1]),
            ),
            (
                "initialize",
                lambda: a.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", ""),
            ),
        ):
            try:
                call()
                check(False, f"{name} after shutdown raises")
            except RuntimeError as e:
                check("shut down" in str(e), f"{name} after shutdown raises ({e})")
        a_ids = daemon.control_clients()
        a = None

        print("== a new engine in the same process")
        b = tcp_engine()
        ok, _ = loopback_write(b, seed=5)
        check(ok, "loopback write with the new engine")
        new = [x for x in daemon.control_clients() if x not in a_ids]
        old_clients = {c for c, _ in a_ids}
        old_tokens = {t for _, t in a_ids}
        check(
            len(new) == 1
            and new[0][0] not in old_clients
            and new[0][1] not in old_tokens
            and new[0][1] != 0,
            "HELLO with a new client id and a new process token",
        )

        print("== a batch left unfreed")
        n = 1 * MiB
        src, dst = HostBuffer(n, 9), HostBuffer(n)
        assert b.register_memory(src.addr, n) == 0
        assert b.register_memory(dst.addr, n) == 0
        session = f"127.0.0.1:{b.get_rpc_port()}"
        bid = b.batch_transfer_async_write(session, [src.addr], [dst.addr], [n])
        check(bid != 0, "async write submitted")
        check(
            wait_until(lambda: b.get_batch_progress(bid)["state"] == "completed"),
            "completed, not freed",
        )
        check(b.shutdown() == 0, "shutdown() frees it and returns 0")
        try:
            b.get_batch_progress(bid)
            check(False, "the batch id is gone")
        except RuntimeError:
            check(True, "the batch id is gone")
        b = None
        check(wait_until(daemon.idle, 2), "the daemon saw every connection close")

        print("== an engine handed out with get_engine()")
        e = tcp_engine()
        inner = e.get_engine()
        try:
            e.shutdown()
            check(False, "shutdown() refused while get_engine()'s engine lives")
        except RuntimeError as err:
            check(
                "still referenced" in str(err),
                f"shutdown() refused while get_engine()'s engine lives ({err})",
            )
        ok, _ = loopback_write(e, seed=7)
        check(ok, "the engine still works after the refusal")
        del inner  # the only other reference
        check(e.shutdown() == 0, "shutdown() once it is released")
        e = None
        check(wait_until(daemon.idle, 2), "the daemon saw every connection close")

        print("== a thread still calling the engine")
        c = tcp_engine()
        ok, (src, dst, session) = loopback_write(c)
        result = {}

        def hammer():
            count = 0
            try:
                while True:
                    c.batch_transfer_sync_write(
                        session, [src.addr], [dst.addr], [src.n]
                    )
                    c.get_rpc_port()
                    count += 1
            except RuntimeError as e:
                result["error"] = str(e)
            result["count"] = count

        th = threading.Thread(target=hammer)
        th.start()
        time.sleep(0.3)
        check(c.shutdown() == 0, "shutdown() while another thread transfers")
        th.join(10)
        check(
            not th.is_alive() and "shut down" in result.get("error", ""),
            f"the thread stopped with RuntimeError after {result.get('count')} writes",
        )
        c = None
        check(wait_until(daemon.idle, 2), "the daemon saw every connection close")
        check(
            resources_settled(before) == before,
            "back to the resources before the engines",
        )
    finally:
        daemon.stop()
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------- GPU mode
class Cuda:
    """The few runtime and driver calls the GPU mode needs (ctypes)."""

    def __init__(self):
        self.rt = ctypes.CDLL("libcudart.so.12")
        self.rt.cudaGetErrorString.restype = ctypes.c_char_p
        self.drv = ctypes.CDLL("libcuda.so.1")
        u64, sz = ctypes.c_ulonglong, ctypes.c_size_t
        d = self.drv
        d.cuMemGetAllocationGranularity.argtypes = [
            ctypes.POINTER(sz),
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        d.cuMemAddressReserve.argtypes = [ctypes.POINTER(u64), sz, sz, u64, u64]
        d.cuMemCreate.argtypes = [ctypes.POINTER(u64), sz, ctypes.c_void_p, u64]
        d.cuMemMap.argtypes = [u64, sz, sz, u64, u64]
        d.cuMemSetAccess.argtypes = [u64, sz, ctypes.c_void_p, sz]
        d.cuMemUnmap.argtypes = [u64, sz]
        d.cuMemRelease.argtypes = [u64]

    def ck(self, rc, what):
        if rc != 0:
            raise RuntimeError(f"{what}: {self.rt.cudaGetErrorString(rc).decode()}")

    def cu(self, rc, what):
        if rc != 0:
            raise RuntimeError(f"{what}: CUresult {rc}")

    def device(self, dev):
        self.ck(self.rt.cudaSetDevice(dev), "cudaSetDevice")
        self.ck(self.rt.cudaFree(None), "context")  # current for driver calls

    def malloc(self, dev, n):
        self.device(dev)
        p = ctypes.c_void_p()
        self.ck(self.rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)), "cudaMalloc")
        return p.value

    def h2d(self, dev, ptr, data):
        self.device(dev)
        self.ck(
            self.rt.cudaMemcpy(
                ctypes.c_void_p(ptr), data, ctypes.c_size_t(len(data)), 1
            ),
            "H2D",
        )

    def d2h(self, dev, ptr, n):
        self.device(dev)
        out = ctypes.create_string_buffer(n)
        self.ck(
            self.rt.cudaMemcpy(out, ctypes.c_void_p(ptr), ctypes.c_size_t(n), 2), "D2H"
        )
        return out.raw


class AllocProp(ctypes.Structure):  # CUmemAllocationProp
    _fields_ = [
        ("type", ctypes.c_int),  # CU_MEM_ALLOCATION_TYPE_PINNED = 1
        ("requested_handle_types", ctypes.c_int),  # POSIX_FILE_DESCRIPTOR = 1
        ("location_type", ctypes.c_int),  # CU_MEM_LOCATION_TYPE_DEVICE = 1
        ("location_id", ctypes.c_int),
        ("win32_metadata", ctypes.c_void_p),
        ("compression_type", ctypes.c_ubyte),
        ("gpu_direct_rdma_capable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class AccessDesc(ctypes.Structure):  # CUmemAccessDesc
    _fields_ = [
        ("location_type", ctypes.c_int),
        ("location_id", ctypes.c_int),
        ("flags", ctypes.c_int),  # CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
    ]


class VmmPool:
    """A releasable pool, as an engine would allocate its KV cache: the
    virtual range stays reserved, the physical allocation (created
    exportable as a POSIX file descriptor) is released and created again."""

    def __init__(self, cuda, dev, n):
        self.cuda, self.dev = cuda, dev
        self.prop = AllocProp(
            type=1, requested_handle_types=1, location_type=1, location_id=dev
        )
        gran = ctypes.c_size_t()
        cuda.device(dev)
        cuda.cu(
            cuda.drv.cuMemGetAllocationGranularity(
                ctypes.byref(gran), ctypes.byref(self.prop), 0
            ),
            "granularity",
        )
        self.size = -(-n // gran.value) * gran.value
        va = ctypes.c_ulonglong()
        cuda.cu(
            cuda.drv.cuMemAddressReserve(ctypes.byref(va), self.size, 0, 0, 0),
            "cuMemAddressReserve",
        )
        self.ptr = va.value
        self.handle = None

    def create(self):
        cuda = self.cuda
        cuda.device(self.dev)
        h = ctypes.c_ulonglong()
        cuda.cu(
            cuda.drv.cuMemCreate(
                ctypes.byref(h), self.size, ctypes.byref(self.prop), 0
            ),
            "cuMemCreate",
        )
        cuda.cu(cuda.drv.cuMemMap(self.ptr, self.size, 0, h.value, 0), "cuMemMap")
        access = AccessDesc(location_type=1, location_id=self.dev, flags=3)
        cuda.cu(
            cuda.drv.cuMemSetAccess(self.ptr, self.size, ctypes.byref(access), 1),
            "cuMemSetAccess",
        )
        self.handle = h.value

    def release(self):
        cuda = self.cuda
        cuda.device(self.dev)
        cuda.cu(cuda.drv.cuMemUnmap(self.ptr, self.size), "cuMemUnmap")
        cuda.cu(cuda.drv.cuMemRelease(self.handle), "cuMemRelease")
        self.handle = None


def gpu_mode(dev0, dev1, vmm, require_vmm):
    sock = os.environ.get("MC_NVLINK_PROXY_SOCKET")
    if not sock or not os.environ.get("MC_NODE_ID"):
        print("--gpu needs MC_NVLINK_PROXY_SOCKET (a running daemon) and MC_NODE_ID")
        FAILURES.append("not configured")
        return
    if require_vmm:
        os.environ["MC_NVLINK_PROXY_REQUIRE_VMM"] = "1"
    elif vmm:
        os.environ["MC_NVLINK_PROXY_VMM"] = "1"
    daemon_bin = shutil.which("mooncake_nvlink_proxy")

    def daemon_stats():
        if not daemon_bin:
            return {}
        out = subprocess.run(
            [daemon_bin, "--socket", sock, "--stats"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        stats = {}
        for kv in out.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                stats[k] = v.split("(")[0]
        return stats

    cuda = Cuda()
    host = os.environ.get("POD_IP", "127.0.0.1")
    n = 64 * MiB
    if vmm:
        src_pool, dst_pool = VmmPool(cuda, dev0, n), VmmPool(cuda, dev1, n)
        src, dst = src_pool.ptr, dst_pool.ptr
    else:
        src, dst = cuda.malloc(dev0, n), cuda.malloc(dev1, n)
    kind = "virtual memory management pools" if vmm else "cudaMalloc blocks"
    for rnd in (1, 2):
        print(f"== round {rnd}: {kind}, written through the daemon, then shut down")
        if vmm:  # new physical memory at the same addresses
            src_pool.create()
            dst_pool.create()
        before = daemon_stats()
        socks_before = socket_inodes()
        target, initiator = TransferEngine(), TransferEngine()
        assert target.initialize(host, "P2PHANDSHAKE", "tcp", "") == 0
        assert initiator.initialize(host, "P2PHANDSHAKE", "tcp", "") == 0
        if require_vmm and rnd == 1:
            plain = cuda.malloc(dev1, MiB)
            check(
                target.register_memory(plain, MiB) != 0,
                "MC_NVLINK_PROXY_REQUIRE_VMM=1 refuses a cudaMalloc block",
            )
        # Round 1: each buffer with its GPU current, as an engine process
        # registers its memory; round 2: both with dev0 current.
        cuda.device(dev1 if rnd == 1 else dev0)
        assert target.register_memory(dst, n) == 0
        cuda.device(dev0)
        assert initiator.register_memory(src, n) == 0
        pattern = bytes((i * (rnd + 2) + 1) & 0xFF for i in range(256)) * (n // 256)
        cuda.h2d(dev0, src, pattern)
        cuda.h2d(dev1, dst, bytes(n))
        session = f"{host}:{target.get_rpc_port()}"
        rc = initiator.batch_transfer_sync_write(session, [src], [dst], [n])
        check(
            rc == 0 and cuda.d2h(dev1, dst, n) == pattern,
            f"round {rnd}: write verified",
        )
        during = daemon_stats()
        if during:
            check(
                int(during.get("copy_requests", 0))
                > int(before.get("copy_requests", 0)),
                f"round {rnd}: the write went through the daemon "
                f"(copy_requests {before.get('copy_requests')} -> "
                f"{during.get('copy_requests')}, vmm_imports="
                f"{during.get('vmm_imports')})",
            )
        check(initiator.shutdown() == 0, f"round {rnd}: initiator shutdown() == 0")
        check(target.shutdown() == 0, f"round {rnd}: target shutdown() == 0")
        after = daemon_stats()
        if after:
            check(
                after.get("ipc_mappings") == before.get("ipc_mappings", "0")
                and after.get("vmm_imports", "0") == before.get("vmm_imports", "0"),
                f"round {rnd}: nothing of this process left in the daemon "
                f"(ipc_mappings={after.get('ipc_mappings')}, "
                f"vmm_imports={after.get('vmm_imports')})",
            )
        left = sockets_opened_since(socks_before)
        check(
            not left,
            f"round {rnd}: no socket the engines opened is left in this process"
            + "".join("\n          " + s for s in left),
        )
        target = initiator = None
        if vmm:  # the pools are released as before a checkpoint
            src_pool.release()
            dst_pool.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--dev0", type=int, default=0)
    ap.add_argument("--dev1", type=int, default=0)
    ap.add_argument("--vmm", action="store_true")
    ap.add_argument("--require-vmm", action="store_true")
    args = ap.parse_args()
    if not sys.platform.startswith("linux"):
        print("skipped: Linux only")
        return 0
    if args.gpu:
        gpu_mode(args.dev0, args.dev1, args.vmm or args.require_vmm, args.require_vmm)
    else:
        cpu_mode()
    print(f"{len(FAILURES)} failure(s)")
    for f in FAILURES:
        print("  - " + f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
