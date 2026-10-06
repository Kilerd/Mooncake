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
    modes = {"target": target, "initiator": initiator, "loop": loop, "concurrent": concurrent}
    sys.exit(modes[sys.argv[1]]() or 0)
