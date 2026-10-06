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
"""Two-process end-to-end test for the nvlink_proxy transport.

Run one "target" and one "initiator" process, typically in two containers on
the same node that each see only their own GPU, plus mooncake_nvlink_proxy
in a container that sees all GPUs. The processes talk over a small TCP side
channel (port 7777) for verification only; data moves through the engine.

  target      allocates a 1 GiB cudaMalloc segment, registers two buffers
              that do NOT start at the segment base (like a framework caching
              allocator hands out), one host buffer, and serves INFO / HASH /
              ZERO / RESET / HHASH requests.
  initiator   writes into the target's buffers through the Python API and
              verifies every byte (sha256) plus untouched guard regions.
  loop        repeated verified writes, for stopping/restarting the daemon
              while transfers run (fallback test).

Environment: POD_IP (session host), DEV (cuda device index, default 0),
TARGET_HOST (initiator/loop), PROTO (initialize protocol, default "tcp"),
LOOP_ITERS / LOOP_SLEEP (loop mode). MC_* variables are passed through to
the engine (MC_NVLINK_PROXY_SOCKET, MC_NODE_ID, MC_FORCE_TCP, ...).
"""

import ctypes
import hashlib
import json
import os
import socket
import statistics
import sys
import time
import zlib

import numpy as np

MiB = 1 << 20
SEG = 1024 * MiB
OFF1, LEN1 = 64 * MiB + 4096, 512 * MiB  # deliberately not segment aligned
OFF2, LEN2 = 700 * MiB, 64 * MiB  # second registration in the same segment
BIG = 281 * MiB  # one large contiguous write
PAGE, NPAGES = 144 * 1024, 2000  # many small page-sized writes
HOST_LEN = 16 * MiB
PORT = 7777

cudart = ctypes.CDLL("libcudart.so.12")
cudart.cudaGetErrorString.restype = ctypes.c_char_p


def ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: {rc} {cudart.cudaGetErrorString(rc).decode()}")


def malloc(n):
    p = ctypes.c_void_p()
    ck(cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)), "cudaMalloc")
    return p.value


def memset0(ptr, n):
    ck(cudart.cudaMemset(ctypes.c_void_p(ptr), 0, ctypes.c_size_t(n)), "cudaMemset")
    ck(cudart.cudaDeviceSynchronize(), "sync")


def h2d(dst, arr):
    ck(cudart.cudaMemcpy(ctypes.c_void_p(dst), arr.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_size_t(arr.nbytes), 1), "H2D")


def d2h(src, n):
    out = np.empty(n, dtype=np.uint8)
    ck(cudart.cudaMemcpy(out.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(src),
                         ctypes.c_size_t(n), 2), "D2H")
    return out


def sha(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def log(*a):
    print(*a, flush=True)


def make_engine():
    from mooncake.engine import TransferEngine

    proto = os.environ.get("PROTO", "tcp")
    host = os.environ["POD_IP"]
    eng = TransferEngine()
    rc = eng.initialize(host, "P2PHANDSHAKE", proto, "")
    env = {k: v for k, v in os.environ.items() if k.startswith("MC_")}
    log(f"INIT proto={proto} rc={rc} env={env}")
    if rc != 0:
        sys.exit(2)
    return eng, f"{host}:{eng.get_rpc_port()}"


def target():
    ck(cudart.cudaSetDevice(int(os.environ.get("DEV", "0"))), "cudaSetDevice")
    eng, session = make_engine()
    seg = malloc(SEG)
    memset0(seg, SEG)
    host = np.zeros(HOST_LEN, dtype=np.uint8)
    reg = {
        "buf1": eng.register_memory(seg + OFF1, LEN1),
        "buf2": eng.register_memory(seg + OFF2, LEN2),
        "host": eng.register_memory(host.ctypes.data, host.nbytes),
    }
    log(f"TARGET session={session} seg={hex(seg)} reg={reg}")
    info = {"session": session, "buf1": seg + OFF1, "buf2": seg + OFF2,
            "host": host.ctypes.data, "reg": reg}
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(8)
    log("READY")
    while True:
        conn, _ = srv.accept()
        with conn:
            cmd = conn.makefile().readline().split()
            if not cmd:
                continue
            if cmd[0] == "INFO":
                out = info
            elif cmd[0] == "HASH":  # offsets relative to the segment base
                out = {"sha": sha(d2h(seg + int(cmd[1]), int(cmd[2])))}
            elif cmd[0] == "HHASH":  # host buffer
                out = {"sha": sha(host[: int(cmd[1])])}
            elif cmd[0] == "ZERO":
                buf = d2h(seg + int(cmd[1]), int(cmd[2]))
                nz = np.flatnonzero(buf)
                out = {"zero": nz.size == 0, "first_nonzero": int(nz[0]) if nz.size else None}
            elif cmd[0] == "RESET":
                memset0(seg, SEG)
                host[:] = 0
                out = {"ok": True}
            else:
                out = {"err": cmd}
            conn.sendall((json.dumps(out) + "\n").encode())


def ask(host, line):
    with socket.create_connection((host, PORT), timeout=120) as s:
        s.sendall((line + "\n").encode())
        return json.loads(s.makefile().readline())


def setup_initiator():
    ck(cudart.cudaSetDevice(int(os.environ.get("DEV", "0"))), "cudaSetDevice")
    th = os.environ["TARGET_HOST"]
    info = ask(th, "INFO")
    log(f"PEER {info}")
    eng, session = make_engine()
    data = np.random.default_rng(1).integers(0, 256, size=LEN1, dtype=np.uint8)
    src = malloc(LEN1)
    h2d(src, data)
    rc = eng.register_memory(src, LEN1)
    log(f"INITIATOR session={session} src_reg={rc}")
    return th, info, eng, data, src


def initiator():
    th, info, eng, data, src = setup_initiator()
    peer, b1, b2 = info["session"], info["buf1"], info["buf2"]
    results = []

    def check(name, ok, extra=""):
        results.append(bool(ok))
        log(f"CHECK {name:<46} {'PASS' if ok else 'FAIL'} {extra}")

    def guards(span_end):
        before = ask(th, f"ZERO 0 {OFF1}")
        after = ask(th, f"ZERO {span_end} {OFF2 - span_end}")
        check("guard: segment base .. buf1 untouched", before["zero"], before)
        check("guard: after written span untouched", after["zero"], after)

    check("target registrations (incl. host) returned 0",
          all(v == 0 for v in info["reg"].values()), info["reg"])

    # A. one large write into a buffer 64 MiB + 4 KiB inside its segment.
    ask(th, "RESET")
    t = time.perf_counter()
    rc = eng.transfer_sync_write(peer, src, b1, BIG)
    dt = time.perf_counter() - t
    log(f"A single-block rc={rc} {dt * 1e3:.2f} ms (first transfer, incl. handle open)")
    check("A: 281 MiB bytes identical", rc == 0 and ask(th, f"HASH {OFF1} {BIG}")["sha"] == sha(data[:BIG]))
    guards(OFF1 + BIG)

    # B. 2000 x 144 KiB pages to permuted destination slots.
    ask(th, "RESET")
    perm = np.random.default_rng(2).permutation(NPAGES)
    srcs = [src + i * PAGE for i in range(NPAGES)]
    dsts = [b1 + int(perm[i]) * PAGE for i in range(NPAGES)]
    t = time.perf_counter()
    rc = eng.batch_transfer_sync_write(peer, srcs, dsts, [PAGE] * NPAGES)
    dt = time.perf_counter() - t
    log(f"B paged rc={rc} {dt * 1e3:.2f} ms")
    expected = data[: NPAGES * PAGE].reshape(NPAGES, PAGE)[np.argsort(perm)]
    check("B: 2000 permuted pages bytes identical", rc == 0 and ask(th, f"HASH {OFF1} {NPAGES * PAGE}")["sha"] == sha(expected))
    guards(OFF1 + NPAGES * PAGE)

    # C. second registration inside the same segment.
    ask(th, "RESET")
    rc = eng.transfer_sync_write(peer, src, b2, 16 * MiB)
    check("C: write into 2nd buffer of same segment", rc == 0 and ask(th, f"HASH {OFF2} {16 * MiB}")["sha"] == sha(data[: 16 * MiB]))
    z = ask(th, f"ZERO 0 {OFF2}")
    check("C: nothing landed before buf2", z["zero"], z)

    # D. read back from the target (READ opcode) into a fresh local buffer.
    dst = malloc(16 * MiB)
    memset0(dst, 16 * MiB)
    rc = eng.register_memory(dst, 16 * MiB)
    rc2 = eng.transfer_sync_read(peer, dst, b2, 16 * MiB)
    check("D: read 16 MiB back from target", rc == 0 and rc2 == 0 and sha(d2h(dst, 16 * MiB)) == sha(data[: 16 * MiB]))

    # E. host memory: registration returns 0 and host->host goes through the base transport.
    hsrc = np.random.default_rng(5).integers(0, 256, size=HOST_LEN, dtype=np.uint8)
    rc = eng.register_memory(hsrc.ctypes.data, hsrc.nbytes)
    rc2 = eng.transfer_sync_write(peer, hsrc.ctypes.data, info["host"], HOST_LEN)
    check("E: host register rc=0 and host->host write", rc == 0 and rc2 == 0 and ask(th, f"HHASH {HOST_LEN}")["sha"] == sha(hsrc))

    # F. bandwidth: 5 x single 281 MiB, 3 x paged.
    single = []
    for _ in range(5):
        t = time.perf_counter()
        assert eng.transfer_sync_write(peer, src, b1, BIG) == 0
        single.append(time.perf_counter() - t)
    paged = []
    for _ in range(3):
        t = time.perf_counter()
        assert eng.batch_transfer_sync_write(peer, srcs, dsts, [PAGE] * NPAGES) == 0
        paged.append(time.perf_counter() - t)
    ms, mp = statistics.median(single), statistics.median(paged)
    log(f"BW single 281MiB median {ms * 1e3:.2f} ms = {BIG / ms / 2**30:.1f} GiB/s; "
        f"paged 2000x144KiB median {mp * 1e3:.2f} ms = {NPAGES * PAGE / mp / 2**30:.1f} GiB/s")
    check("F: data still exact after bandwidth runs", ask(th, f"HASH {OFF1} {NPAGES * PAGE}")["sha"] == sha(expected))
    log(f"SUMMARY {'ALL PASS' if all(results) else 'FAILURES'} ({sum(results)}/{len(results)})")
    time.sleep(float(os.environ.get("FINAL_SLEEP", "2")))  # let the stats line flush
    return 0 if all(results) else 1


def loop():
    """Verified writes with a different payload each iteration."""
    th, info, eng, data, src = setup_initiator()
    peer, b1 = info["session"], info["buf1"]
    n = 32 * MiB
    iters = int(os.environ.get("LOOP_ITERS", "60"))
    pause = float(os.environ.get("LOOP_SLEEP", "0.5"))
    ok = 0
    for i in range(iters):
        off = (i * 1048576 + i * 4096) % (LEN1 - n)
        t = time.perf_counter()
        rc = eng.transfer_sync_write(peer, src + off, b1, n)
        dt = time.perf_counter() - t
        good = rc == 0 and ask(th, f"HASH {OFF1} {n}")["sha"] == sha(data[off:off + n])
        ok += good
        log(f"LOOP {i:3d} rc={rc} {dt * 1e3:7.2f} ms {n / dt / 2**30:6.1f} GiB/s {'PASS' if good else 'FAIL'}")
        time.sleep(pause)
    log(f"LOOP SUMMARY {ok}/{iters} {'ALL PASS' if ok == iters else 'FAILURES'}")
    time.sleep(float(os.environ.get("FINAL_SLEEP", "2")))
    return 0 if ok == iters else 1


# --------------------------------------------------------------------------
# Many-small-entry benchmark: NBLK separately allocated blocks per side (like
# per-layer K/V tensors of a page-granular KV cache) and batches of NENT
# entries of about SLOT bytes at random, mostly non-adjacent slots.
NBLK = int(os.environ.get("NBLK", "72"))
SLOTS = int(os.environ.get("SLOTS", "2048"))
SLOT = int(os.environ.get("SLOT", "6144"))
NENT = int(os.environ.get("NENT", "7660"))
BLK = SLOTS * SLOT


def mtarget():
    ck(cudart.cudaSetDevice(int(os.environ.get("DEV", "0"))), "cudaSetDevice")
    eng, session = make_engine()
    blocks = [malloc(BLK) for _ in range(NBLK)]
    for b in blocks:
        memset0(b, BLK)
    regs = [eng.register_memory(b, BLK) for b in blocks]
    log(f"MTARGET session={session} blocks={NBLK}x{BLK} reg_ok={all(r == 0 for r in regs)}")
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(8)
    log("READY")
    while True:
        conn, _ = srv.accept()
        with conn:
            cmd = conn.makefile().readline().split()
            if not cmd:
                continue
            if cmd[0] == "MINFO":
                out = {"session": session, "blocks": blocks, "reg": regs}
            elif cmd[0] == "MHASH":
                out = {"sha": [sha(d2h(b, BLK)) for b in blocks]}
            elif cmd[0] == "MRESET":
                for b in blocks:
                    memset0(b, BLK)
                out = {"ok": True}
            else:
                out = {"err": cmd}
            conn.sendall((json.dumps(out) + "\n").encode())


def daemon_stats():
    """Counters of the daemon behind MC_NVLINK_PROXY_SOCKET (or {})."""
    import struct

    path = os.environ.get("MC_NVLINK_PROXY_SOCKET")
    if not path:
        return {}
    try:
        with socket.socket(socket.AF_UNIX) as s:
            s.settimeout(5)
            s.connect(path)
            s.sendall(struct.pack("<IHHIIQ", 0x504E434D, 1, 6, 0, 0, 1))
            hdr = b""
            while len(hdr) < 24:
                hdr += s.recv(24 - len(hdr))
            plen = struct.unpack("<IHHIIQ", hdr)[4]
            body = b""
            while len(body) < plen:
                body += s.recv(plen - len(body))
        return {k: v for k, v in (kv.split("=", 1) for kv in body.decode().split()) if v.isdigit()}
    except OSError:
        return {}


def _pattern(name, rng):
    """Returns (src_blk, src_off, dst_blk, dst_off, length) arrays."""
    per = [NENT // NBLK + (1 if i < NENT % NBLK else 0) for i in range(NBLK)]
    rows = []
    if name.startswith("size"):
        size = int(name[4:])
        slots = BLK // size
        n = max(1, (NENT * SLOT) // size)
        per = [n // NBLK + (1 if i < n % NBLK else 0) for i in range(NBLK)]
        for b in range(NBLK):
            k = min(per[b], slots)
            for s_, d_ in zip(rng.choice(slots, k, replace=False), rng.choice(slots, k, replace=False)):
                rows.append((b, int(s_) * size, b, int(d_) * size, size))
        return rows
    adj = 0.5 if name == "runs50" else 0.0
    for b in range(NBLK):
        used_s, used_d = set(), set()
        prev = None
        while len(used_d) < per[b]:
            if prev and rng.random() < adj and prev[0] + 1 < SLOTS and prev[1] + 1 < SLOTS \
                    and prev[0] + 1 not in used_s and prev[1] + 1 not in used_d:
                s_, d_ = prev[0] + 1, prev[1] + 1
            else:
                s_ = int(rng.integers(SLOTS))
                d_ = int(rng.integers(SLOTS))
                if s_ in used_s or d_ in used_d:
                    continue
            used_s.add(s_)
            used_d.add(d_)
            prev = (s_, d_)
            if name == "misaligned":
                so, do = int(rng.integers(16)), int(rng.integers(16))
                rows.append((b, s_ * SLOT + so, b, d_ * SLOT + do, SLOT - 16))
            else:
                rows.append((b, s_ * SLOT, b, d_ * SLOT, SLOT))
    if name == "interleaved":
        order = rng.permutation(len(rows))
        rows = [rows[i] for i in order]
    return rows


def smallbench():
    ck(cudart.cudaSetDevice(int(os.environ.get("DEV", "0"))), "cudaSetDevice")
    th = os.environ["TARGET_HOST"]
    info = ask(th, "MINFO")
    eng, session = make_engine()
    rng = np.random.default_rng(7)
    src_host = [rng.integers(0, 256, size=BLK, dtype=np.uint8) for _ in range(NBLK)]
    src = []
    for h in src_host:
        p = malloc(BLK)
        h2d(p, h)
        src.append(p)
    regs = [eng.register_memory(p, BLK) for p in src]
    log(f"SMALLBENCH session={session} blocks={NBLK}x{BLK} reg_ok={all(r == 0 for r in regs)} "
        f"target_reg_ok={all(r == 0 for r in info['reg'])}")
    peer, dblk = info["session"], info["blocks"]
    reps = int(os.environ.get("REPS", "10"))
    patterns = os.environ.get("PATTERNS", "random,runs50,interleaved,misaligned,size4096,size16384,size65536,size262144").split(",")
    results = []
    for name in patterns:
        rows = _pattern(name, np.random.default_rng(zlib.crc32(name.encode())))
        srcs = [src[b] + o for b, o, _, _, _ in rows]
        dsts = [dblk[b] + o for _, _, b, o, _ in rows]
        lens = [n for *_, n in rows]
        nbytes = sum(lens)
        ask(th, "MRESET")
        for _ in range(2):  # warm-up (handle opens, table allocation)
            assert eng.batch_transfer_sync_write(peer, srcs, dsts, lens) == 0
        d0 = daemon_stats()
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            rc = eng.batch_transfer_sync_write(peer, srcs, dsts, lens)
            times.append(time.perf_counter() - t)
            assert rc == 0, rc
        d1 = daemon_stats()
        expected = [np.zeros(BLK, dtype=np.uint8) for _ in range(NBLK)]
        for (sb, so, db, do, n) in rows:
            expected[db][do:do + n] = src_host[sb][so:so + n]
        got = ask(th, "MHASH")["sha"]
        ok = got == [sha(e) for e in expected]
        results.append(ok)
        med = statistics.median(times)
        dd = {k: int(d1.get(k, 0)) - int(d0.get(k, 0)) for k in ("copy_requests", "copy_us_total", "plan_us_total", "exec_us_total", "coalesced_entries", "kernel_entries", "ce_entries")}
        reqs = max(1, dd["copy_requests"])
        daemon = (f" daemon/batch: total {dd['copy_us_total'] / reqs / 1e3:.2f} ms plan {dd['plan_us_total'] / reqs / 1e3:.2f} "
                  f"exec {dd['exec_us_total'] / reqs / 1e3:.2f} merged {dd['coalesced_entries'] // reqs} "
                  f"kernel {dd['kernel_entries'] // reqs} ce {dd['ce_entries'] // reqs}") if d1 else ""
        log(f"BENCH {name:<12} entries={len(rows):5d} bytes={nbytes / 2**20:7.1f}MiB median {med * 1e3:7.2f} ms "
            f"(min {min(times) * 1e3:.2f} max {max(times) * 1e3:.2f}) = {nbytes / med / 2**30:6.1f} GiB/s "
            f"verify {'PASS' if ok else 'FAIL'}{daemon}")
    log(f"SUMMARY {'ALL PASS' if all(results) else 'FAILURES'} ({sum(results)}/{len(results)})")
    time.sleep(float(os.environ.get("FINAL_SLEEP", "1")))
    return 0 if all(results) else 1


# --------------------------------------------------------------------------
# Continuous stress: THREADS submitters, each sending batches of ~SNENT
# entries of SSLOT bytes into its own region of SNBLK blocks, with a
# different source mapping every round and a byte-exact check of the
# thread's destination slots after every batch. Meant to run while the copy
# daemon is killed / restarted or peers are paused.
STHREADS = int(os.environ.get("THREADS", "8"))
SNBLK = int(os.environ.get("SNBLK", "72"))
SSLOT = int(os.environ.get("SSLOT", "2560"))
SNENT = int(os.environ.get("SNENT", "20000"))
SREGION = int(os.environ.get("SREGION", "512"))  # slots per thread per block
SBLK = STHREADS * SREGION * SSLOT


def _stress_dst_slots(t):
    """Fixed destination slots of thread t: list over blocks of sorted arrays."""
    per = [SNENT // SNBLK + (1 if b < SNENT % SNBLK else 0) for b in range(SNBLK)]
    return [np.sort(np.random.default_rng(1000003 * t + b).choice(SREGION, per[b], replace=False))
            for b in range(SNBLK)]


def _stress_src_slots(t, r, dst):
    rng = np.random.default_rng((t + 1) * 7919 + r * 104729)
    return [rng.choice(SREGION, len(d), replace=False) for d in dst]


def starget():
    import socketserver

    ck(cudart.cudaSetDevice(int(os.environ.get("DEV", "0"))), "cudaSetDevice")
    eng, session = make_engine()
    blocks = [malloc(SBLK) for _ in range(SNBLK)]
    for b in blocks:
        memset0(b, SBLK)
    regs = [eng.register_memory(b, SBLK) for b in blocks]
    dst_slots = {t: _stress_dst_slots(t) for t in range(STHREADS)}
    log(f"STARGET session={session} blocks={SNBLK}x{SBLK} reg_ok={all(r == 0 for r in regs)}")

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            cmd = self.rfile.readline().decode().split()
            if not cmd:
                return
            if cmd[0] == "SINFO":
                out = {"session": session, "blocks": blocks, "reg": regs}
            elif cmd[0] == "THASH":  # destination slots of one thread, in order
                t = int(cmd[1])
                h = hashlib.sha256()
                for b, d in enumerate(dst_slots[t]):
                    region = d2h(blocks[b] + t * SREGION * SSLOT, SREGION * SSLOT)
                    h.update(region.reshape(SREGION, SSLOT)[d].tobytes())
                out = {"sha": h.hexdigest()}
            else:
                out = {"err": cmd}
            self.wfile.write((json.dumps(out) + "\n").encode())

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    log("READY")
    Server(("0.0.0.0", PORT), Handler).serve_forever()


def stress():
    import threading

    ck(cudart.cudaSetDevice(int(os.environ.get("DEV", "0"))), "cudaSetDevice")
    th = os.environ["TARGET_HOST"]
    info = ask(th, "SINFO")
    eng, session = make_engine()
    rng = np.random.default_rng(11)
    src_host = [rng.integers(0, 256, size=SBLK, dtype=np.uint8) for _ in range(SNBLK)]
    src = []
    for h in src_host:
        p = malloc(SBLK)
        h2d(p, h)
        src.append(p)
    regs = [eng.register_memory(p, SBLK) for p in src]
    peer, dblk = info["session"], info["blocks"]
    duration = float(os.environ.get("DURATION", "60"))
    log(f"STRESS session={session} threads={STHREADS} entries/batch={SNENT} x {SSLOT}B "
        f"blocks={SNBLK} duration={duration}s reg_ok={all(r == 0 for r in regs)} "
        f"target_reg_ok={all(r == 0 for r in info['reg'])}")
    stats = {"batches": 0, "failed": 0, "mismatch": 0, "lat": []}
    lock = threading.Lock()
    t_end = time.time() + duration
    t0 = time.time()

    def worker(t):
        dst = _stress_dst_slots(t)
        dsts = [dblk[b] + (t * SREGION + int(x)) * SSLOT for b, d in enumerate(dst) for x in d]
        r = 0
        while time.time() < t_end:
            srcs_slots = _stress_src_slots(t, r, dst)
            srcs = [src[b] + (t * SREGION + int(x)) * SSLOT for b, s in enumerate(srcs_slots) for x in s]
            ts = time.time()
            rc = eng.batch_transfer_sync_write(peer, srcs, dsts, [SSLOT] * len(srcs))
            dt = time.time() - ts
            h = hashlib.sha256()
            for b, s in enumerate(srcs_slots):
                h.update(src_host[b][t * SREGION * SSLOT:(t + 1) * SREGION * SSLOT]
                         .reshape(SREGION, SSLOT)[s].tobytes())
            ok = rc == 0 and ask(th, f"THASH {t}")["sha"] == h.hexdigest()
            with lock:
                stats["batches"] += 1
                stats["lat"].append(dt)
                if rc != 0:
                    stats["failed"] += 1
                elif not ok:
                    stats["mismatch"] += 1
            if rc != 0 or not ok or dt > 1.0:
                log(f"EVENT t={time.time() - t0:6.1f}s thread={t} round={r} rc={rc} "
                    f"{'MISMATCH' if rc == 0 and not ok else ''} transfer {dt * 1e3:.0f} ms")
            r += 1

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(STHREADS)]
    for x in threads:
        x.start()
    last = 0
    while any(x.is_alive() for x in threads):
        time.sleep(1)
        el = time.time() - t0
        if el - last >= 5:
            last = el
            with lock:
                lat = sorted(stats["lat"][-200:]) or [0]
                log(f"PROGRESS t={el:5.1f}s batches={stats['batches']} failed={stats['failed']} "
                    f"mismatch={stats['mismatch']} recent p50={lat[len(lat) // 2] * 1e3:.1f} ms "
                    f"p99={lat[int(len(lat) * 0.99)] * 1e3:.1f} ms")
    for x in threads:
        x.join()
    lat = sorted(stats["lat"]) or [0]
    good = stats["failed"] == 0 and stats["mismatch"] == 0 and stats["batches"] > 0
    log(f"STRESS SUMMARY batches={stats['batches']} failed={stats['failed']} "
        f"mismatch={stats['mismatch']} p50={lat[len(lat) // 2] * 1e3:.1f} ms "
        f"p99={lat[int(len(lat) * 0.99)] * 1e3:.1f} ms max={lat[-1] * 1e3:.0f} ms "
        f"{'ALL PASS' if good else 'FAILURES'}")
    time.sleep(float(os.environ.get("FINAL_SLEEP", "6")))
    return 0 if good else 1


def concurrent():
    """Several threads write disjoint regions at the same time."""
    import threading

    th, info, eng, data, src = setup_initiator()
    peer, b1 = info["session"], info["buf1"]
    threads_n = int(os.environ.get("THREADS", "8"))
    iters = int(os.environ.get("ITERS", "20"))
    pages = (LEN1 // threads_n) // PAGE
    chunk = pages * PAGE  # contiguous per-thread regions
    failures = []
    ask(th, "RESET")

    def worker(t):
        base = t * chunk
        srcs = [src + base + i * PAGE for i in range(pages)]
        dsts = [b1 + base + i * PAGE for i in range(pages)]
        for _ in range(iters):
            if eng.batch_transfer_sync_write(peer, srcs, dsts, [PAGE] * pages) != 0:
                failures.append(t)

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(t,)) for t in range(threads_n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    total = threads_n * iters * pages * PAGE
    span = threads_n * pages * PAGE
    good = not failures and ask(th, f"HASH {OFF1} {span}")["sha"] == sha(data[:span])
    log(f"CONCURRENT {threads_n} threads x {iters} batches x {pages} pages: {dt * 1e3:.1f} ms "
        f"{total / dt / 2**30:.1f} GiB/s failures={len(failures)} {'PASS' if good else 'FAIL'}")
    time.sleep(float(os.environ.get("FINAL_SLEEP", "2")))
    return 0 if good else 1


if __name__ == "__main__":
    modes = {"target": target, "initiator": initiator, "loop": loop, "concurrent": concurrent,
             "mtarget": mtarget, "smallbench": smallbench,
             "starget": starget, "stress": stress}
    sys.exit(modes[sys.argv[1]]() or 0)
