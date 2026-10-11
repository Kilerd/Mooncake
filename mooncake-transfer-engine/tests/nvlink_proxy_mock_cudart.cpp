// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// CPU stand-in for the CUDA runtime calls of mooncake_nvlink_proxy, linked
// instead of libcudart to run the daemon without GPUs
// (nvlink_proxy_cpu_test.py). Device memory is host memory.
//
// - Streams are asynchronous: every stream has a worker thread that runs its
//   copies, kernel launches and event records in order; synchronize, event
//   waits and stream-waits-event behave like CUDA's.
// - IPC handles name POSIX shared memory segments ("GPU memory" exported by
//   the test's engines): "MOCKSHM\0", int32 GPU, NUL-terminated name (at most
//   43 bytes), uint32 exporter pid at byte 56. As the driver does, a process
//   gets one mapping per (GPU context, handle): opening a handle that is
//   already open there returns the same mapping (MOCK_IPC_REOPEN=error makes
//   it fail instead), and a close unmaps it for every user. An exporter pid
//   equal to the importer's fails the open (error 201).
// - Copies between two GPUs follow MOCK_P2P: ok (default), drop (writes
//   lost) or corrupt (one byte per KiB flipped); MOCK_BAD_PAIRS="a>b,..."
//   limits that to these ordered pairs; MOCK_NOPEER="a>b,..." reports no
//   peer access.
// - MOCK_COPY_DELAY_MS: each copy that touches IPC-imported memory takes
//   this long (on its stream). MOCK_HOST_DELAY_MS: each copy of more than
//   64 KiB from/to pinned host memory takes this long.
// - MOCK_WRITE_LOG=<file>: one line per copy into IPC-imported memory,
//   "<CLOCK_MONOTONIC us> <segment name> <offset> <length>", written when
//   the copy lands.
// - MOCK_GPUS (default 8), MOCK_OOM_GPU=k, MOCK_NO_CONTEXT_GPU=k,
//   MOCK_NOKERNEL=1 (the gather kernel has no image), MOCK_POOL_BAD=1
//   (copies from/to the registered staging pool are dropped).

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>

#include "gather_copy.h"

namespace {

std::mutex mu;                                       // guards the maps below
std::map<uintptr_t, std::pair<size_t, int>> allocs;  // base -> (size, dev)
std::map<uintptr_t, std::string> imported;           // base -> segment name
std::set<std::pair<int, int>> peers;                 // (dev, peer)
std::map<uintptr_t, size_t> pinned;                  // cudaHostAlloc ranges
std::map<uintptr_t, size_t> registered;              // cudaHostRegister ranges
// (dev, handle bytes) -> mapping base: one mapping per context and handle.
std::map<std::pair<int, std::string>, uintptr_t> ipc_open;
thread_local int cur = 0;

int envInt(const char *n, int d) {
    const char *v = getenv(n);
    return v && *v ? atoi(v) : d;
}
int ngpus() { return envInt("MOCK_GPUS", 8); }

bool inRanges(const std::map<uintptr_t, size_t> &m, const void *p) {
    std::lock_guard<std::mutex> l(mu);
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    auto it = m.upper_bound(a);
    if (it == m.begin()) return false;
    --it;
    return a < it->first + it->second;
}
bool isPinned(const void *p) {
    return inRanges(pinned, p) || inRanges(registered, p);
}

// Owning GPU of device memory (-1: host memory), and for IPC-imported
// memory its segment name and offset.
int devOf(const void *p, std::string *segment = nullptr,
          uint64_t *offset = nullptr) {
    std::lock_guard<std::mutex> l(mu);
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    auto it = allocs.upper_bound(a);
    if (it == allocs.begin()) return -1;
    --it;
    if (a >= it->first + it->second.first) return -1;
    if (segment) {
        auto im = imported.find(it->first);
        if (im != imported.end()) {
            *segment = im->second;
            if (offset) *offset = a - it->first;
        }
    }
    return it->second.second;
}

bool inList(const char *env, int s, int d) {
    const char *v = getenv(env);
    if (!v || !*v) return false;
    const std::string all = std::string(",") + v + ",";
    return all.find("," + std::to_string(s) + ">" + std::to_string(d) + ",") !=
           std::string::npos;
}
bool badPair(int s, int d) {
    const char *v = getenv("MOCK_BAD_PAIRS");
    return !v || !*v || inList("MOCK_BAD_PAIRS", s, d);
}

uint64_t monoUs() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000 + uint64_t(ts.tv_nsec) / 1000;
}

void logWrite(const std::string &segment, uint64_t off, size_t n) {
    static std::mutex log_mu;
    const char *path = getenv("MOCK_WRITE_LOG");
    if (!path || !*path) return;
    std::lock_guard<std::mutex> l(log_mu);
    FILE *f = fopen(path, "a");
    if (!f) return;
    fprintf(f, "%llu %s %llu %zu\n", (unsigned long long)monoUs(),
            segment.c_str(), (unsigned long long)off, n);
    fclose(f);
}

// One copy, executed where it lands (a stream's worker, or the caller for
// synchronous calls).
cudaError_t copyNow(void *dst, const void *src, size_t n) {
    std::string seg_s, seg_d;
    uint64_t off_d = 0;
    const int sd = devOf(src, &seg_s), dd = devOf(dst, &seg_d, &off_d);
    if (envInt("MOCK_POOL_BAD", 0) &&
        (inRanges(registered, dst) || inRanges(registered, src)))
        return cudaSuccess;
    const int host_delay = envInt("MOCK_HOST_DELAY_MS", 0);
    if (host_delay && (isPinned(dst) || isPinned(src)) && n > 65536)
        usleep(host_delay * 1000);
    const int delay = envInt("MOCK_COPY_DELAY_MS", 0);
    if (delay && (!seg_s.empty() || !seg_d.empty())) usleep(delay * 1000);
    if (sd >= 0 && dd >= 0 && sd != dd && badPair(sd, dd)) {
        const char *mode = getenv("MOCK_P2P");
        const std::string m = mode ? mode : "ok";
        if (m == "drop") return cudaSuccess;
        if (m == "corrupt") {
            memcpy(dst, src, n);
            for (size_t i = 0; i < n; i += 1024)
                static_cast<uint8_t *>(dst)[i] ^= 0x5a;
            if (!seg_d.empty()) logWrite(seg_d, off_d, n);
            return cudaSuccess;
        }
    }
    memcpy(dst, src, n);
    if (!seg_d.empty()) logWrite(seg_d, off_d, n);
    return cudaSuccess;
}

// ----------------------------------------------------------------- streams
struct Stream {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::function<void()>> q;
    bool busy = false;
    std::thread th;
    Stream() {
        th = std::thread([this]() {
            while (true) {
                std::function<void()> op;
                {
                    std::unique_lock<std::mutex> l(m);
                    cv.wait(l, [this]() { return !q.empty(); });
                    op = std::move(q.front());
                    q.pop_front();
                    busy = true;
                }
                op();
                {
                    std::lock_guard<std::mutex> l(m);
                    busy = false;
                }
                cv.notify_all();
            }
        });
        th.detach();
    }
    void push(std::function<void()> op) {
        {
            std::lock_guard<std::mutex> l(m);
            q.push_back(std::move(op));
        }
        cv.notify_all();
    }
    void sync() {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [this]() { return q.empty() && !busy; });
    }
};

Stream *streamOf(cudaStream_t s) {
    // The legacy / per-thread default streams run synchronously here.
    if (s == nullptr || s == cudaStreamLegacy || s == cudaStreamPerThread)
        return nullptr;
    return reinterpret_cast<Stream *>(s);
}

void submit(cudaStream_t s, std::function<void()> op) {
    Stream *st = streamOf(s);
    if (st)
        st->push(std::move(op));
    else
        op();
}

struct Event {
    std::mutex m;
    std::condition_variable cv;
    uint64_t recorded = 0, done = 0;
    void complete(uint64_t gen) {
        {
            std::lock_guard<std::mutex> l(m);
            if (gen > done) done = gen;
        }
        cv.notify_all();
    }
    void waitFor(uint64_t gen) {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&]() { return done >= gen; });
    }
};

}  // namespace

extern "C" {

cudaError_t cudaGetDeviceCount(int *n) {
    *n = ngpus();
    return *n > 0 ? cudaSuccess : cudaErrorNoDevice;
}

static cudaError_t fillProps(cudaDeviceProp *p, int d) {
    memset(p, 0, sizeof(*p));
    snprintf(p->name, sizeof(p->name), "Mock GPU");
    for (int i = 0; i < 16; ++i)
        p->uuid.bytes[i] = static_cast<char>(d * 16 + i);
    p->pciDomainID = 0;
    p->pciBusID = 0x10 * (d + 1);
    p->pciDeviceID = 0;
    return cudaSuccess;
}
#if CUDART_VERSION >= 12000
cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp *p, int d) {
    return fillProps(p, d);
}
#endif
#if CUDART_VERSION >= 13000 || !defined(cudaGetDeviceProperties)
#undef cudaGetDeviceProperties
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int d) {
    return fillProps(p, d);
}
#endif

cudaError_t cudaSetDevice(int d) {
    if (d < 0 || d >= ngpus()) return cudaErrorInvalidDevice;
    cur = d;
    return cudaSuccess;
}

cudaError_t cudaFree(void *p) {
    if (!p) {  // context creation
        if (envInt("MOCK_NO_CONTEXT_GPU", -1) == cur)
            return cudaErrorMemoryAllocation;
        return cudaSuccess;
    }
    std::lock_guard<std::mutex> l(mu);
    allocs.erase(reinterpret_cast<uintptr_t>(p));
    free(p);
    return cudaSuccess;
}

cudaError_t cudaMalloc(void **p, size_t n) {
    if (envInt("MOCK_OOM_GPU", -1) == cur) return cudaErrorMemoryAllocation;
    *p = malloc(n);
    std::lock_guard<std::mutex> l(mu);
    allocs[reinterpret_cast<uintptr_t>(*p)] = {n, cur};
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void *d, const void *s, size_t n, cudaMemcpyKind) {
    return copyNow(d, s, n);
}

cudaError_t cudaMemcpyAsync(void *d, const void *s, size_t n, cudaMemcpyKind,
                            cudaStream_t stream) {
    submit(stream, [=]() { copyNow(d, s, n); });
    return cudaSuccess;
}

#if CUDART_VERSION >= 13000
cudaError_t cudaMemcpyBatchAsync(void *const *dsts, const void *const *srcs,
                                 const size_t *sizes, size_t count,
                                 cudaMemcpyAttributes *, size_t *, size_t,
                                 cudaStream_t stream) {
#else
cudaError_t cudaMemcpyBatchAsync(void **dsts, void **srcs, size_t *sizes,
                                 size_t count, cudaMemcpyAttributes *, size_t *,
                                 size_t, size_t *, cudaStream_t stream) {
#endif
    std::vector<void *> d(count), s(count);
    std::vector<size_t> n(sizes, sizes + count);
    for (size_t i = 0; i < count; ++i) {
        d[i] = const_cast<void *>(static_cast<const void *>(dsts[i]));
        s[i] = const_cast<void *>(static_cast<const void *>(srcs[i]));
    }
    submit(stream, [d, s, n]() {
        for (size_t i = 0; i < d.size(); ++i) copyNow(d[i], s[i], n[i]);
    });
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    Stream *st = streamOf(s);
    if (st) st->sync();
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *s, unsigned int) {
    *s = reinterpret_cast<cudaStream_t>(new Stream());
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t s) {
    Stream *st = streamOf(s);
    if (st) st->sync();  // the worker thread stays parked (test process)
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int *v, cudaDeviceAttr a, int) {
    *v = a == cudaDevAttrComputeMode ? cudaComputeModeDefault : 1;
    return cudaSuccess;
}

cudaError_t cudaHostAlloc(void **p, size_t n, unsigned int) {
    *p = malloc(n);
    std::lock_guard<std::mutex> l(mu);
    pinned[reinterpret_cast<uintptr_t>(*p)] = n;
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void *p) {
    {
        std::lock_guard<std::mutex> l(mu);
        pinned.erase(reinterpret_cast<uintptr_t>(p));
    }
    free(p);
    return cudaSuccess;
}

cudaError_t cudaHostRegister(void *p, size_t n, unsigned int) {
    std::lock_guard<std::mutex> l(mu);
    registered[reinterpret_cast<uintptr_t>(p)] = n;
    return cudaSuccess;
}

cudaError_t cudaHostUnregister(void *p) {
    std::lock_guard<std::mutex> l(mu);
    registered.erase(reinterpret_cast<uintptr_t>(p));
    return cudaSuccess;
}

cudaError_t cudaHostGetDevicePointer(void **d, void *h, unsigned int) {
    *d = h;
    return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *e, unsigned int) {
    *e = reinterpret_cast<cudaEvent_t>(new Event());
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s) {
    Event *ev = reinterpret_cast<Event *>(e);
    uint64_t gen;
    {
        std::lock_guard<std::mutex> l(ev->m);
        gen = ++ev->recorded;
    }
    submit(s, [ev, gen]() { ev->complete(gen); });
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    Event *ev = reinterpret_cast<Event *>(e);
    uint64_t gen;
    {
        std::lock_guard<std::mutex> l(ev->m);
        gen = ev->recorded;
    }
    ev->waitFor(gen);
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t e) {
    delete reinterpret_cast<Event *>(e);
    return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned int) {
    Event *ev = reinterpret_cast<Event *>(e);
    uint64_t gen;
    {
        std::lock_guard<std::mutex> l(ev->m);
        gen = ev->recorded;
    }
    submit(s, [ev, gen]() { ev->waitFor(gen); });
    return cudaSuccess;
}

cudaError_t cudaGetLastError(void) { return cudaSuccess; }

const char *cudaGetErrorString(cudaError_t e) {
    switch (e) {
        case cudaSuccess:
            return "no error";
        case cudaErrorMemoryAllocation:
            return "out of memory (mock)";
        case cudaErrorDeviceUninitialized:
            return "invalid device context (mock)";
        case cudaErrorAlreadyMapped:
            return "resource already mapped (mock)";
        case cudaErrorInvalidResourceHandle:
            return "invalid resource handle (mock)";
        default:
            return "mock error";
    }
}

cudaError_t cudaDeviceCanAccessPeer(int *c, int dev, int peer) {
    *c = inList("MOCK_NOPEER", dev, peer) ? 0 : 1;
    return cudaSuccess;
}

cudaError_t cudaDeviceEnablePeerAccess(int peer, unsigned int) {
    std::lock_guard<std::mutex> l(mu);
    if (!peers.insert({cur, peer}).second)
        return cudaErrorPeerAccessAlreadyEnabled;
    return cudaSuccess;
}

cudaError_t cudaDeviceReset(void) {
    std::lock_guard<std::mutex> l(mu);
    for (auto it = peers.begin(); it != peers.end();)
        it = (it->first == cur || it->second == cur) ? peers.erase(it)
                                                     : std::next(it);
    return cudaSuccess;
}

cudaError_t cudaIpcOpenMemHandle(void **p, cudaIpcMemHandle_t h, unsigned int) {
    const char *b = h.reserved;
    if (memcmp(b, "MOCKSHM", 8) != 0) return cudaErrorInvalidResourceHandle;
    int dev;
    memcpy(&dev, b + 8, 4);
    char name[48] = {};
    memcpy(name, b + 12, 43);
    uint32_t exporter_pid = 0;
    memcpy(&exporter_pid, b + 56, 4);
    if (exporter_pid != 0 && exporter_pid == uint32_t(getpid()))
        return cudaErrorDeviceUninitialized;  // 201, as the driver
    const std::pair<int, std::string> key{cur, std::string(b, 64)};
    {
        std::lock_guard<std::mutex> l(mu);
        auto it = ipc_open.find(key);
        if (it != ipc_open.end()) {
            const char *mode = getenv("MOCK_IPC_REOPEN");
            if (mode && std::string(mode) == "error")
                return cudaErrorAlreadyMapped;
            *p = reinterpret_cast<void *>(it->second);
            return cudaSuccess;
        }
    }
    const int fd = shm_open(name, O_RDWR, 0);
    if (fd < 0) return cudaErrorInvalidResourceHandle;
    struct stat st;
    fstat(fd, &st);
    void *m =
        mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return cudaErrorInvalidResourceHandle;
    std::lock_guard<std::mutex> l(mu);
    allocs[reinterpret_cast<uintptr_t>(m)] = {size_t(st.st_size), dev};
    imported[reinterpret_cast<uintptr_t>(m)] = name;
    ipc_open[key] = reinterpret_cast<uintptr_t>(m);
    *p = m;
    return cudaSuccess;
}

cudaError_t cudaIpcCloseMemHandle(void *p) {
    std::lock_guard<std::mutex> l(mu);
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    auto it = allocs.find(a);
    if (it == allocs.end() || !imported.count(a))
        return cudaErrorInvalidResourceHandle;
    munmap(p, it->second.first);  // for every user of the mapping
    allocs.erase(it);
    imported.erase(a);
    for (auto o = ipc_open.begin(); o != ipc_open.end();)
        o = o->second == a ? ipc_open.erase(o) : std::next(o);
    return cudaSuccess;
}

}  // extern "C"

// Gather kernel stand-in (same contract as gather_copy.cu): reads the entry
// table when it runs on the stream.
namespace mooncake {
namespace nvlink_proxy {
cudaError_t launchGatherCopy(const GatherEntry *entries, uint32_t count, int,
                             cudaStream_t stream) {
    if (envInt("MOCK_NOKERNEL", 0)) return cudaErrorNoKernelImageForDevice;
    submit(stream, [entries, count]() {
        for (uint32_t i = 0; i < count; ++i)
            copyNow(reinterpret_cast<void *>(entries[i].dst),
                    reinterpret_cast<const void *>(entries[i].src),
                    entries[i].length);
    });
    return cudaSuccess;
}
}  // namespace nvlink_proxy
}  // namespace mooncake
