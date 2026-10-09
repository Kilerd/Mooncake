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

// mooncake_nvlink_proxy: node-local GPU copy daemon for the nvlink_proxy
// transport. See nvlink_proxy_protocol.h for the wire protocol and
// docs/source/design/transfer-engine/nvlink-proxy.md for deployment.
//
// The daemon must see every GPU of the node (e.g. a container without a GPU
// request and NVIDIA_VISIBLE_DEVICES=all). At start it enables peer access
// between every GPU pair; it then opens the clients' legacy CUDA IPC handles
// lazily in the context of the GPU that receives the data and copies with
// that GPU's copy engines over NVLink / PCIe P2P.
//
// Before it serves copies between two GPUs it has a child process copy known
// patterns between them in both directions (P2P self-test, --p2p-selftest):
// some platforms report working peer access yet drop or corrupt the data. A
// pair that fails is denied, as are pairs named with --deny-peer /
// --deny-pair; copies between the GPUs of a denied pair are refused with
// kNoPeerAccess and the engines move them to their base transport (see
// p2p_policy.h).

#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <system_error>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gather_copy.h"
#include "p2p_policy.h"
#include "transport/nvlink_proxy_transport/nvlink_proxy_protocol.h"

using namespace mooncake::nvlink_proxy;

extern char **environ;

namespace {

// ----------------------------------------------------------------- logging
std::mutex g_log_mu;

void logf(const char *level, const char *fmt, ...) {
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);
    char when[32];
    strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", &tm);
    std::lock_guard<std::mutex> lock(g_log_mu);
    fprintf(stderr, "%s.%03ld %s nvlink_proxy: %s\n", when,
            ts.tv_nsec / 1000000, level, msg);
    fflush(stderr);
}

#define LOGI(...) logf("I", __VA_ARGS__)
#define LOGW(...) logf("W", __VA_ARGS__)
#define LOGE(...) logf("E", __VA_ARGS__)

uint64_t nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string hexUuid(const uint8_t *u) {
    char buf[64];
    snprintf(buf, sizeof(buf),
             "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%"
             "02x%02x",
             u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10],
             u[11], u[12], u[13], u[14], u[15]);
    return buf;
}

// ------------------------------------------------------------- socket I/O
bool writeAll(int fd, const void *buf, size_t len) {
    const char *p = static_cast<const char *>(buf);
    while (len > 0) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

bool readAll(int fd, void *buf, size_t len) {
    char *p = static_cast<char *>(buf);
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n == 0) return false;
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

bool sendMsg(int fd, MsgType type, uint64_t seq, ProxyStatus status,
             const void *payload, size_t len) {
    MsgHeader h{};
    h.magic = kMagic;
    h.version = kProtocolVersion;
    h.type = static_cast<uint16_t>(type);
    h.status = static_cast<uint32_t>(status);
    h.payload_len = static_cast<uint32_t>(len);
    h.seq = seq;
    std::string buf(reinterpret_cast<const char *>(&h), sizeof(h));
    if (len) buf.append(static_cast<const char *>(payload), len);
    return writeAll(fd, buf.data(), buf.size());
}

// --------------------------------------------------------------- devices
struct Device {
    int ordinal = -1;
    uint8_t uuid[kUuidSize];
    std::string name;
    std::string pci;
    PciAddr addr;
    bool exclusive = false;  // compute mode not Default
    int numa = 0;            // NUMA node of the GPU (sysfs; 0 if unknown)
};

constexpr int kMaxDevices = 64;
std::vector<Device> g_devices;
// g_peer[dst][src]: dst's context has peer access to src's memory.
std::atomic<bool> g_peer[kMaxDevices][kMaxDevices];
// A GPU gets a CUDA context (and peer access to the other initialized GPUs)
// only when the first block on it is registered, unless --eager-init.
std::mutex g_dev_mu;
std::atomic<bool> g_dev_ready[kMaxDevices];
bool ensureDevice(int dev);
uint64_t g_epoch = 0;
uint64_t g_start_us = 0;

// P2P policy (p2p_policy.h): --p2p-selftest and the pairs denied by the
// command line or by the self-test, indexed by daemon ordinal.
SelfTestMode g_selftest = SelfTestMode::kEnforce;
PeerPolicy g_policy;
std::vector<std::string> g_pci_short;  // ordinal -> "bb:dd" for stats
// Per pair (both orders hold the same value): not tested yet (or the last
// test could not run: tested again at the next registration on either GPU),
// passed, failed (final until the daemon restarts).
enum : uint8_t {
    kNotTested = 0,
    kTestPassed = 1,
    kTestFailed = 2,
};
std::atomic<uint8_t> g_test_state[kMaxDevices][kMaxDevices];
// The self-test runs in a child process (this binary, re-executed) that is
// killed when it makes no progress for this long.
int g_selftest_timeout_s = 30;
std::string g_self_exe;  // /proc/self/exe
// GPU pairs (and per-GPU host paths) whose last test could not run, waiting
// for a retry (--stats).
std::atomic<size_t> g_untested_pairs{0};
std::atomic<size_t> g_untested_host_paths{0};
std::atomic<size_t> g_untested_staged_pairs{0};
void requestSelfTestForDevice(int dev);

bool isStickyCudaError(cudaError_t e) {
    switch (e) {
        case cudaErrorIllegalAddress:
        case cudaErrorLaunchFailure:
        case cudaErrorHardwareStackError:
        case cudaErrorIllegalInstruction:
        case cudaErrorMisalignedAddress:
        case cudaErrorInvalidAddressSpace:
        case cudaErrorInvalidPc:
        case cudaErrorECCUncorrectable:
        case cudaErrorUnknown:
            return true;
        default:
            return false;
    }
}

[[noreturn]] void dieOnStickyError(cudaError_t e, const char *what) {
    LOGE(
        "unrecoverable CUDA error in %s: %s; exiting so the daemon can be "
        "restarted (clients fall back to their base transport meanwhile)",
        what, cudaGetErrorString(e));
    _exit(70);
}

// ----------------------------------------------------------- registrations
struct Registration {
    uint64_t client_id = 0;
    uint64_t base = 0;
    uint64_t size = 0;
    int device = -1;  // daemon ordinal of the GPU that owns the memory
    uint8_t handle[kIpcHandleSize];
    // Control connection that currently owns this registration (guarded by
    // g_reg_mu). A client that reconnects re-registers its keys and takes
    // ownership, so the old connection's cleanup must not drop them.
    uint64_t owner_conn = 0;

    std::mutex mu;
    std::unordered_map<int, char *> mapped;  // pulling device -> mapped base

    ~Registration() {
        for (auto &kv : mapped) {
            cudaSetDevice(kv.first);
            cudaError_t e = cudaIpcCloseMemHandle(kv.second);
            if (e != cudaSuccess)
                LOGW("cudaIpcCloseMemHandle(client=%016" PRIx64
                     " base=%#" PRIx64 " dev=%d): %s",
                     client_id, base, kv.first, cudaGetErrorString(e));
        }
    }

    // Map this block in |dev|'s context (the GPU that will pull/receive).
    cudaError_t mapFor(int dev, char **out, bool *opened) {
        std::lock_guard<std::mutex> lock(mu);
        *opened = false;
        auto it = mapped.find(dev);
        if (it != mapped.end()) {
            *out = it->second;
            return cudaSuccess;
        }
        cudaError_t e = cudaSetDevice(dev);
        if (e != cudaSuccess) return e;
        cudaIpcMemHandle_t h;
        memcpy(&h, handle, sizeof(h));
        void *ptr = nullptr;
        uint64_t t0 = nowUs();
        e = cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess);
        if (e != cudaSuccess) return e;
        LOGI("opened client=%016" PRIx64 " base=%#" PRIx64 " size=%" PRIu64
             " (gpu %d) in gpu %d context in %.1f ms",
             client_id, base, size, device, dev, (nowUs() - t0) / 1000.0);
        mapped[dev] = static_cast<char *>(ptr);
        *out = static_cast<char *>(ptr);
        *opened = true;
        return cudaSuccess;
    }
};

using RegKey = std::pair<uint64_t, uint64_t>;  // (client_id, base)
std::mutex g_reg_mu;
std::map<RegKey, std::shared_ptr<Registration>> g_regs;

// ------------------------------------------------------------------ stats
std::atomic<bool> g_stop{false};
// COPY requests being executed; the daemon waits for them on shutdown.
std::atomic<int> g_inflight_copies{0};

struct Stats {
    std::atomic<uint64_t> connections{0};
    std::atomic<uint64_t> active_connections{0};
    std::atomic<uint64_t> active_clients{0};
    std::atomic<uint64_t> registrations{0};
    std::atomic<uint64_t> copy_requests{0};
    std::atomic<uint64_t> copy_entries{0};
    std::atomic<uint64_t> copy_bytes{0};
    std::atomic<uint64_t> copy_failures{0};
    std::atomic<uint64_t> copy_us_total{0};
    std::atomic<uint64_t> copy_us_max{0};
    std::atomic<uint64_t> handle_opens{0};
    std::atomic<uint64_t> coalesced_entries{0};  // merged into a neighbour
    std::atomic<uint64_t> kernel_entries{0};
    std::atomic<uint64_t> kernel_bytes{0};
    std::atomic<uint64_t> ce_entries{0};
    std::atomic<uint64_t> ce_bytes{0};
    std::atomic<uint64_t> plan_us_total{0};
    std::atomic<uint64_t> exec_us_total{0};
    std::atomic<uint64_t> denied_copies{0};  // refused by the P2P policy
    std::atomic<uint64_t> staged_copies{0};  // served through host memory
    std::atomic<uint64_t> staged_bytes{0};
    std::atomic<uint64_t> staged_failures{0};  // then refused (base transport)
} g_stats;

void updateMax(std::atomic<uint64_t> &m, uint64_t v) {
    uint64_t cur = m.load();
    while (v > cur && !m.compare_exchange_weak(cur, v)) {
    }
}

std::string statsText() {
    size_t regs;
    {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        regs = g_regs.size();
    }
    uint64_t reqs = g_stats.copy_requests.load();
    uint64_t us = g_stats.copy_us_total.load();
    char buf[1536];
    snprintf(
        buf, sizeof(buf),
        "epoch=%016" PRIx64 " uptime_s=%" PRIu64
        " devices=%zu active_clients=%" PRIu64 " active_connections=%" PRIu64
        " registrations=%zu copy_requests=%" PRIu64 " copy_entries=%" PRIu64
        " copy_bytes=%" PRIu64 " copy_failures=%" PRIu64 " avg_copy_us=%" PRIu64
        " max_copy_us=%" PRIu64 " avg_plan_us=%" PRIu64 " avg_exec_us=%" PRIu64
        " copy_us_total=%" PRIu64 " plan_us_total=%" PRIu64
        " exec_us_total=%" PRIu64 " coalesced_entries=%" PRIu64
        " kernel_entries=%" PRIu64 " kernel_bytes=%" PRIu64
        " ce_entries=%" PRIu64 " ce_bytes=%" PRIu64 " handle_opens=%" PRIu64,
        g_epoch, (nowUs() - g_start_us) / 1000000, g_devices.size(),
        g_stats.active_clients.load(), g_stats.active_connections.load(), regs,
        reqs, g_stats.copy_entries.load(), g_stats.copy_bytes.load(),
        g_stats.copy_failures.load(), reqs ? us / reqs : 0,
        g_stats.copy_us_max.load(),
        reqs ? g_stats.plan_us_total.load() / reqs : 0,
        reqs ? g_stats.exec_us_total.load() / reqs : 0, us,
        g_stats.plan_us_total.load(), g_stats.exec_us_total.load(),
        g_stats.coalesced_entries.load(), g_stats.kernel_entries.load(),
        g_stats.kernel_bytes.load(), g_stats.ce_entries.load(),
        g_stats.ce_bytes.load(), g_stats.handle_opens.load());
    std::string out = buf;
    const std::string denied = g_policy.deniedList(g_pci_short);
    out += " p2p_selftest=";
    out += selfTestModeName(g_selftest);
    out += " denied_pairs=" + std::to_string(g_policy.deniedCount());
    if (!denied.empty()) out += "(" + denied + ")";
    out +=
        " denied_copy_requests=" + std::to_string(g_stats.denied_copies.load());
    out +=
        " untested_pairs=" + std::to_string(g_untested_pairs.load()) +
        " untested_host_paths=" + std::to_string(g_untested_host_paths.load()) +
        " untested_staged_pairs=" +
        std::to_string(g_untested_staged_pairs.load());
    out += " staged_copies=" + std::to_string(g_stats.staged_copies.load()) +
           " staged_bytes=" + std::to_string(g_stats.staged_bytes.load()) +
           " staged_failures=" + std::to_string(g_stats.staged_failures.load());
    return out;
}

// ------------------------------------------------------------ copy engine
// Tunables (command line).
struct CopyConfig {
    // Entries shorter than this go through the gather/scatter kernel, longer
    // ones through the copy engines (cudaMemcpyBatchAsync). 0 = never use
    // the kernel.
    uint64_t gather_threshold = 128 * 1024;
    // Which GPU runs the kernel: the one receiving the data (reads through
    // the peer mapping) or the one holding the source (writes through it).
    bool gather_on_src = false;
    // Merge consecutive entries that are contiguous in source and target.
    bool coalesce = true;
} g_cfg;

// Per connection-thread resources, one set per executing GPU.
struct DeviceResources {
    cudaStream_t stream = nullptr;
    GatherEntry *host_table = nullptr;  // pinned
    GatherEntry *dev_table = nullptr;
    size_t capacity = 0;
    int grid_blocks = 0;
};

struct ThreadResources {
    std::unordered_map<int, DeviceResources> devs;
    ~ThreadResources() {
        for (auto &kv : devs) {
            cudaSetDevice(kv.first);
            if (kv.second.stream) cudaStreamDestroy(kv.second.stream);
            if (kv.second.dev_table) cudaFree(kv.second.dev_table);
            if (kv.second.host_table) cudaFreeHost(kv.second.host_table);
        }
    }
    cudaError_t get(int dev, DeviceResources **out) {
        auto &r = devs[dev];
        *out = &r;
        if (r.stream) return cudaSuccess;
        cudaError_t e = cudaSetDevice(dev);
        if (e != cudaSuccess) return e;
        e = cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking);
        if (e != cudaSuccess) {
            r.stream = nullptr;
            return e;
        }
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        r.grid_blocks = std::max(1, sms) * 16;
        return cudaSuccess;
    }
    cudaError_t reserve(int dev, DeviceResources &r, size_t n) {
        if (n <= r.capacity) return cudaSuccess;
        size_t cap =
            std::max<size_t>(n, std::max<size_t>(1024, r.capacity * 2));
        cudaError_t e = cudaSetDevice(dev);
        if (e != cudaSuccess) return e;
        if (r.dev_table) cudaFree(r.dev_table);
        if (r.host_table) cudaFreeHost(r.host_table);
        r.dev_table = nullptr;
        r.host_table = nullptr;
        r.capacity = 0;
        e = cudaMalloc(&r.dev_table, cap * sizeof(GatherEntry));
        if (e != cudaSuccess) return e;
        e = cudaHostAlloc(&r.host_table, cap * sizeof(GatherEntry),
                          cudaHostAllocDefault);
        if (e != cudaSuccess) return e;
        r.capacity = cap;
        return cudaSuccess;
    }
};

thread_local ThreadResources t_res;
std::atomic<bool> g_batch_api_usable{true};
std::atomic<bool> g_kernel_unusable[kMaxDevices];

cudaError_t issueCopies(std::vector<void *> &dsts, std::vector<void *> &srcs,
                        std::vector<size_t> &sizes, cudaStream_t stream) {
    const size_t count = dsts.size();
    if (count == 0) return cudaSuccess;
#if CUDART_VERSION >= 12080
    if (g_batch_api_usable.load(std::memory_order_relaxed)) {
        // srcAccessOrderStream: the source is read in stream order, which is
        // what a peer copy needs; matches IntraNodeNvlinkTransport.
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        size_t attr_idx = 0;
        constexpr size_t kChunk = 8192;
        for (size_t off = 0; off < count; off += kChunk) {
            size_t n = std::min(kChunk, count - off);
#if CUDART_VERSION >= 13000
            cudaError_t e = cudaMemcpyBatchAsync(
                const_cast<const void **>(dsts.data() + off),
                const_cast<const void **>(srcs.data() + off),
                sizes.data() + off, n, &attr, &attr_idx, 1, stream);
#else
            size_t fail_idx = n;
            cudaError_t e = cudaMemcpyBatchAsync(
                dsts.data() + off, srcs.data() + off, sizes.data() + off, n,
                &attr, &attr_idx, 1, &fail_idx, stream);
#endif
            if (e == cudaErrorNotSupported ||
                e == cudaErrorCallRequiresNewerDriver) {
                // Older driver: fall back to per-copy submission for good.
                cudaGetLastError();
                if (off != 0) return e;  // partially issued; let caller fail
                g_batch_api_usable.store(false);
                LOGW(
                    "cudaMemcpyBatchAsync unavailable (%s); using "
                    "cudaMemcpyAsync per entry",
                    cudaGetErrorString(e));
                break;
            }
            if (e != cudaSuccess) return e;
            if (off + n == count) return cudaSuccess;
        }
    }
#endif
    for (size_t i = 0; i < count; ++i) {
        cudaError_t e = cudaMemcpyAsync(dsts[i], srcs[i], sizes[i],
                                        cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

// ------------------------------------------------------- host staging
// --p2p-fallback=host (default): a copy between two GPUs that may not use
// P2P -- a refused pair (self-test failed, not verified yet in enforce mode,
// denied on the command line) or a pair without peer access -- goes through
// pinned host memory in this daemon instead of being refused: the sending
// GPU copies into a host slot (copy engines for large runs, the gather kernel
// packing small rows), the receiving GPU copies out of it (copy engines, the
// same kernel scattering the rows), two slots alternating so that piece k+1
// travels to the host while piece k travels on. Only for a pair whose GPUs
// passed the host-path self-test and whose staged copy passed the pair's
// self-test (the real stagedCopy() in both directions, byte-verified). Any
// error refuses the copy as before, and the engine uses its base transport.
// The pinned pool (--host-staging-mb, split over the NUMA nodes with GPUs)
// is reserved and touched at start by a thread bound to each node's CPUs
// (node-local pages; a memory limit too small shows at start) and pinned with
// cudaHostRegister when first used.
bool g_fallback_host = true;
uint64_t g_host_staging_bytes = 128ull << 20;
int g_host_staging_wait_ms = 100;
// Host-path self-test per GPU, staged-copy self-test per pair [min][max]:
// kNotTested / kTestPassed / kTestFailed.
std::atomic<uint8_t> g_host_state[kMaxDevices];
std::atomic<uint8_t> g_stage_state[kMaxDevices][kMaxDevices];

bool hostStageable(int src, int dst) {
    const int a = std::min(src, dst), b = std::max(src, dst);
    return g_fallback_host && g_host_state[src].load() == kTestPassed &&
           g_host_state[dst].load() == kTestPassed &&
           g_stage_state[a][b].load() == kTestPassed;
}

std::string readSysfs(const std::string &path) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return "";
    char buf[4096] = {};
    const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    std::string out(buf, n);
    while (!out.empty() && (out.back() == '\n' || out.back() == ' '))
        out.pop_back();
    return out;
}

int readGpuNumaNode(const PciAddr &a) {
    char path[128];
    snprintf(path, sizeof(path),
             "/sys/bus/pci/devices/%04x:%02x:%02x.0/numa_node", a.domain, a.bus,
             a.device);
    const std::string v = readSysfs(path);
    return v.empty() ? 0 : std::max(0, atoi(v.c_str()));
}

int gpuNumaNode(int dev) { return g_devices[dev].numa; }

size_t gpuNumaNodes() {
    std::set<int> nodes;
    for (const Device &d : g_devices) nodes.insert(d.numa);
    return std::max<size_t>(1, nodes.size());
}

// "0-79,160-239" -> CPU set; false when unreadable or empty.
bool nodeCpus(int node, cpu_set_t *set) {
    const std::string list = readSysfs("/sys/devices/system/node/node" +
                                       std::to_string(node) + "/cpulist");
    CPU_ZERO(set);
    size_t i = 0;
    int count = 0;
    while (i < list.size()) {
        size_t j = list.find(',', i);
        if (j == std::string::npos) j = list.size();
        const std::string r = list.substr(i, j - i);
        const size_t dash = r.find('-');
        const int lo = atoi(r.c_str());
        const int hi =
            dash == std::string::npos ? lo : atoi(r.c_str() + dash + 1);
        for (int c = lo; c <= hi && c < CPU_SETSIZE; ++c) {
            CPU_SET(c, set);
            ++count;
        }
        i = j + 1;
    }
    return count > 0;
}

// Runs |fn| on a short-lived thread (to bind it to a node's CPUs without
// touching the caller); false when no thread could be started.
bool runOnThread(const std::function<void()> &fn) {
    try {
        std::thread(fn).join();
        return true;
    } catch (const std::system_error &e) {
        LOGE("cannot start a thread: %s", e.what());
        return false;
    }
}

class StagingPool {
   public:
    explicit StagingPool(int node) : node_(node) {}

    uint64_t slotBytes() const { return slot_bytes_; }

    // Reserves and touches the pool's memory from a thread bound to the
    // node's CPUs (first touch: node-local pages). At start, in the
    // background; again on first use if that failed.
    bool reserve(std::string *err) {
        std::lock_guard<std::mutex> lock(mu_);
        return reserveLocked(err);
    }

    // Two slots, both or none (no copy ever holds one while waiting for
    // another, so concurrent copies cannot deadlock). False when the pool
    // cannot be set up or |deadline_us| passes first.
    bool acquire(int dev, char **a, char **b, uint64_t deadline_us,
                 std::string *err) {
        std::unique_lock<std::mutex> lock(mu_);
        if (!pinned_ && !pinLocked(dev, err)) return false;
        while (free_.size() < 2) {
            const uint64_t now = nowUs();
            if (now >= deadline_us) {
                *err = "host staging pool busy";
                return false;
            }
            cv_.wait_for(lock, std::chrono::microseconds(deadline_us - now));
        }
        *a = free_.back();
        free_.pop_back();
        *b = free_.back();
        free_.pop_back();
        return true;
    }

    void release(char *a, char *b) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            free_.push_back(a);
            free_.push_back(b);
        }
        cv_.notify_all();
    }

   private:
    bool reserveLocked(std::string *err) {
        if (mem_) return true;
        size_t slots = 0;
        if (!stagingGeometry(g_host_staging_bytes, gpuNumaNodes(), &slot_bytes_,
                             &slots)) {
            *err =
                "--host-staging-mb leaves fewer than two slots per NUMA node";
            return false;
        }
        bytes_ = slot_bytes_ * slots;
        void *p = nullptr;
        bool bound = false;
        const bool ran = runOnThread([&]() {
            cpu_set_t set;
            if (nodeCpus(node_, &set))
                bound = sched_setaffinity(0, sizeof(set), &set) == 0;
            if (posix_memalign(&p, 2 << 20, bytes_) == 0)
                memset(p, 0, bytes_);  // touch: resident and node-local now
            else
                p = nullptr;
        });
        if (!ran || !p) {
            *err = "cannot reserve " + std::to_string(bytes_ >> 20) +
                   " MiB of host memory for staging";
            return false;
        }
        mem_ = static_cast<char *>(p);
        for (size_t i = 0; i < slots; ++i)
            free_.push_back(mem_ + i * slot_bytes_);
        LOGI("host staging pool on NUMA node %d: %zu x %" PRIu64
             " MiB reserved%s",
             node_, slots, slot_bytes_ >> 20,
             bound ? "" : " (not bound to the node's CPUs)");
        return true;
    }

    bool pinLocked(int dev, std::string *err) {
        const uint64_t now = nowUs();
        if (failed_at_us_ && now - failed_at_us_ < 60000000) {
            *err = error_;
            return false;
        }
        cudaError_t e = cudaSuccess;
        if (!reserveLocked(&error_)) {
            e = cudaErrorMemoryAllocation;
        } else {
            e = cudaSetDevice(dev);
            if (e == cudaSuccess)
                e = cudaHostRegister(
                    mem_, bytes_,
                    cudaHostRegisterPortable | cudaHostRegisterMapped);
            void *dp = nullptr;
            if (e == cudaSuccess) e = cudaHostGetDevicePointer(&dp, mem_, 0);
            if (e == cudaSuccess && dp != mem_) {
                // Kernels address the slots by their host address.
                cudaHostUnregister(mem_);
                e = cudaErrorNotSupported;
            }
            if (e != cudaSuccess) {
                cudaGetLastError();
                error_ = std::string("pinning the staging pool: ") +
                         cudaGetErrorString(e);
            }
        }
        if (e != cudaSuccess) {
            failed_at_us_ = now;
            *err = error_;
            LOGW("host staging on NUMA node %d unavailable: %s", node_,
                 error_.c_str());
            return false;
        }
        pinned_ = true;
        LOGI("host staging pool on NUMA node %d pinned (%" PRIu64 " MiB)",
             node_, bytes_ >> 20);
        return true;
    }

    const int node_;
    std::mutex mu_;
    std::condition_variable cv_;
    char *mem_ = nullptr;
    uint64_t bytes_ = 0, slot_bytes_ = 0;
    bool pinned_ = false;
    std::vector<char *> free_;
    uint64_t failed_at_us_ = 0;
    std::string error_;
};

StagingPool &stagingPool(int node) {
    static std::mutex mu;
    static std::map<int, std::unique_ptr<StagingPool>> pools;
    std::lock_guard<std::mutex> lock(mu);
    auto &p = pools[node];
    if (!p) p.reset(new StagingPool(node));
    return *p;
}

struct StagedItem {
    char *src;  // mapped in the sending GPU's context
    char *dst;  // mapped in the receiving GPU's context
    uint64_t len;
};

// Copies |items| from GPU |s| to GPU |d| through host memory. Waits for
// staging slots until |wait_until_us|; stops issuing pieces once
// |hard_deadline_us| has passed (the engine has given up on the copy by
// then) and drains what is in flight before returning. False with *err set:
// the caller refuses the copy (some pieces may have been written; the base
// transport rewrites them all).
bool stagedCopy(int s, int d, const std::vector<StagedItem> &items,
                uint64_t wait_until_us, uint64_t hard_deadline_us,
                std::string *err) {
    StagingPool &pool = stagingPool(gpuNumaNode(s));
    char *slot[2] = {nullptr, nullptr};
    if (!pool.acquire(s, &slot[0], &slot[1],
                      std::min(wait_until_us, hard_deadline_us), err))
        return false;
    struct Piece {
        char *src;
        char *dst;
        uint64_t len;
        uint64_t off;  // in the slot, 16-byte aligned
    };
    const uint64_t cap = pool.slotBytes();
    std::vector<std::vector<Piece>> chunks(1);
    uint64_t used = 0;
    for (const StagedItem &it : items) {
        uint64_t done = 0;
        while (done < it.len) {
            used = alignSlotOffset(used);
            if (used >= cap) {
                chunks.emplace_back();
                used = 0;
            }
            const uint64_t n = std::min(it.len - done, cap - used);
            chunks.back().push_back({it.src + done, it.dst + done, n, used});
            used += n;
            done += n;
        }
    }
    const uint64_t thr = g_cfg.gather_threshold;
    auto small = [&](uint64_t len, int dev) {
        return len < thr && !g_kernel_unusable[dev].load();
    };
    size_t max_small = 0;
    for (const auto &c : chunks) {
        size_t n = 0;
        for (const Piece &p : c) n += small(p.len, s) || small(p.len, d);
        max_small = std::max(max_small, n);
    }
    DeviceResources *rs = nullptr, *rd = nullptr;
    cudaEvent_t filled[2] = {nullptr, nullptr}, freed[2] = {nullptr, nullptr};
    cudaError_t e = t_res.get(s, &rs);
    if (e == cudaSuccess) e = t_res.get(d, &rd);
    if (e == cudaSuccess && max_small) e = t_res.reserve(s, *rs, 2 * max_small);
    if (e == cudaSuccess && max_small) e = t_res.reserve(d, *rd, 2 * max_small);
    for (int k = 0; k < 2 && e == cudaSuccess; ++k) {
        e = cudaSetDevice(s);
        if (e == cudaSuccess)
            e = cudaEventCreateWithFlags(&filled[k], cudaEventDisableTiming);
        if (e == cudaSuccess) e = cudaSetDevice(d);
        if (e == cudaSuccess)
            e = cudaEventCreateWithFlags(&freed[k], cudaEventDisableTiming);
    }
    std::string why;
    // One side of a piece list: copy engines for large pieces, one gather
    // kernel launch (table region |region|) for the small ones -- or the
    // copy engines for those too when the kernel has no image for |dev|.
    auto side = [&](int dev, DeviceResources *res, size_t region,
                    const std::vector<Piece> &c, char *host,
                    bool to_host) -> cudaError_t {
        const cudaMemcpyKind kind =
            to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        cudaError_t r = cudaSetDevice(dev);
        std::vector<GatherEntry> table;
        for (const Piece &p : c) {
            if (r != cudaSuccess) break;
            char *from = to_host ? p.src : host + p.off;
            char *to = to_host ? host + p.off : p.dst;
            if (small(p.len, dev))
                table.push_back({reinterpret_cast<uint64_t>(to),
                                 reinterpret_cast<uint64_t>(from), p.len});
            else
                r = cudaMemcpyAsync(to, from, p.len, kind, res->stream);
        }
        if (r == cudaSuccess && !table.empty()) {
            GatherEntry *ht = res->host_table + region * max_small;
            GatherEntry *dt = res->dev_table + region * max_small;
            memcpy(ht, table.data(), table.size() * sizeof(GatherEntry));
            r = cudaMemcpyAsync(dt, ht, table.size() * sizeof(GatherEntry),
                                cudaMemcpyHostToDevice, res->stream);
            if (r == cudaSuccess)
                r = launchGatherCopy(dt, static_cast<uint32_t>(table.size()),
                                     res->grid_blocks, res->stream);
            if (r == cudaErrorNoKernelImageForDevice ||
                r == cudaErrorInvalidDeviceFunction ||
                r == cudaErrorUnsupportedPtxVersion) {
                // As on the P2P path: copy engines from now on.
                cudaGetLastError();
                g_kernel_unusable[dev] = true;
                LOGW(
                    "gather kernel unavailable on gpu %d; small copies use "
                    "the copy engines",
                    dev);
                r = cudaSuccess;
                for (const GatherEntry &g : table) {
                    r = cudaMemcpyAsync(reinterpret_cast<void *>(g.dst),
                                        reinterpret_cast<void *>(g.src),
                                        g.length, kind, res->stream);
                    if (r != cudaSuccess) break;
                }
            }
        }
        return r;
    };
    bool slot_used[2] = {false, false};
    for (size_t k = 0; k < chunks.size() && e == cudaSuccess; ++k) {
        if (nowUs() >= hard_deadline_us) {
            why = "deadline reached after " + std::to_string(k) + " of " +
                  std::to_string(chunks.size()) + " pieces";
            break;
        }
        const int sl = int(k % 2);
        // The slot (and the table regions of its parity) are free again once
        // the receiving GPU has copied piece k-2 out of it.
        if (slot_used[sl]) e = cudaEventSynchronize(freed[sl]);
        if (e == cudaSuccess) e = side(s, rs, sl, chunks[k], slot[sl], true);
        if (e == cudaSuccess) e = cudaEventRecord(filled[sl], rs->stream);
        if (e == cudaSuccess && nowUs() >= hard_deadline_us) {
            // Nothing of piece k reaches the destination.
            why = "deadline reached after " + std::to_string(k) + " of " +
                  std::to_string(chunks.size()) + " pieces";
            break;
        }
        if (e == cudaSuccess) e = cudaSetDevice(d);
        if (e == cudaSuccess)
            e = cudaStreamWaitEvent(rd->stream, filled[sl], 0);
        if (e == cudaSuccess) e = side(d, rd, sl, chunks[k], slot[sl], false);
        if (e == cudaSuccess) e = cudaEventRecord(freed[sl], rd->stream);
        slot_used[sl] = true;
    }
    // Nothing may still run when the slots go back or the reply goes out.
    cudaError_t se = rd ? cudaStreamSynchronize(rd->stream) : cudaSuccess;
    if (e == cudaSuccess) e = se;
    se = rs ? cudaStreamSynchronize(rs->stream) : cudaSuccess;
    if (e == cudaSuccess) e = se;
    for (int k = 0; k < 2; ++k) {
        if (filled[k]) cudaEventDestroy(filled[k]);
        if (freed[k]) cudaEventDestroy(freed[k]);
    }
    pool.release(slot[0], slot[1]);
    if (e != cudaSuccess) {
        if (isStickyCudaError(e)) dieOnStickyError(e, "host-staged copy");
        cudaGetLastError();
        *err = std::string("CUDA error: ") + cudaGetErrorString(e);
        return false;
    }
    if (!why.empty()) {
        *err = why;
        return false;
    }
    return true;
}

struct CopyResult {
    ProxyStatus status = ProxyStatus::kOk;
    uint32_t failed_index = 0;
    uint64_t bytes = 0;
    std::string message;
    bool denied = false;  // refused by the P2P policy, not a failure
};

std::shared_ptr<Registration> lookupReg(uint64_t client, uint64_t base) {
    std::lock_guard<std::mutex> lock(g_reg_mu);
    auto it = g_regs.find({client, base});
    return it == g_regs.end() ? nullptr : it->second;
}

struct RegKeyHash {
    size_t operator()(const RegKey &k) const {
        return std::hash<uint64_t>()(k.first * 0x9e3779b97f4a7c15ULL ^
                                     k.second);
    }
};

// Resolves (client, base) keys of one request, keeping every registration it
// references alive until the copies completed even if the owner disconnects.
class RegResolver {
   public:
    Registration *find(uint64_t client, uint64_t base) {
        if (last_ && last_->client_id == client && last_->base == base)
            return last_;
        if (last2_ && last2_->client_id == client && last2_->base == base) {
            std::swap(last_, last2_);
            return last_;
        }
        auto it = cache_.find({client, base});
        Registration *reg;
        if (it != cache_.end()) {
            reg = it->second;
        } else {
            auto sp = lookupReg(client, base);
            if (!sp) return nullptr;
            reg = sp.get();
            keep_.push_back(std::move(sp));
            cache_.emplace(RegKey{client, base}, reg);
        }
        last2_ = last_;
        last_ = reg;
        return reg;
    }

   private:
    Registration *last_ = nullptr, *last2_ = nullptr;
    std::unordered_map<RegKey, Registration *, RegKeyHash> cache_;
    std::vector<std::shared_ptr<Registration>> keep_;
};

// Caches the mapping of a block in an executing GPU's context.
class MapResolver {
   public:
    cudaError_t map(Registration *reg, int dev, char **out) {
        for (auto &h : hot_) {
            if (h.reg == reg && h.dev == dev) {
                *out = h.ptr;
                return cudaSuccess;
            }
        }
        bool opened = false;
        cudaError_t e = reg->mapFor(dev, out, &opened);
        if (opened) g_stats.handle_opens++;
        if (e != cudaSuccess) return e;
        hot_[next_] = {reg, dev, *out};
        next_ = (next_ + 1) % kHot;
        return cudaSuccess;
    }

   private:
    static constexpr int kHot = 4;
    struct Hot {
        Registration *reg = nullptr;
        int dev = -1;
        char *ptr = nullptr;
    } hot_[kHot];
    int next_ = 0;
};

CopyResult doCopy(const CopyEntry *entries, uint32_t count, uint32_t budget_ms,
                  uint64_t t_recv_us) {
    CopyResult r;
    r.failed_index = count;
    auto fail = [&](ProxyStatus st, uint32_t idx, std::string msg) {
        r.status = st;
        r.failed_index = idx;
        r.message = std::move(msg);
        return r;
    };

    // 1. Resolve and validate every entry; merge runs that are contiguous in
    //    both source and destination (page-granular KV caches produce many).
    struct Item {
        Registration *src;
        Registration *dst;
        uint64_t src_off;
        uint64_t dst_off;
        uint64_t len;
        bool staged;  // refused pair, copied through host memory
    };
    std::vector<Item> items;
    items.reserve(count);
    RegResolver regs;
    for (uint32_t i = 0; i < count; ++i) {
        const CopyEntry &e = entries[i];
        if (e.length == 0) continue;
        Registration *src = regs.find(e.src_client, e.src_base);
        if (!src) {
            char m[128];
            snprintf(m, sizeof(m),
                     "source block client=%016" PRIx64 " base=%#" PRIx64
                     " is not registered",
                     e.src_client, e.src_base);
            return fail(ProxyStatus::kUnknownSource, i, m);
        }
        Registration *dst = regs.find(e.dst_client, e.dst_base);
        if (!dst) {
            char m[128];
            snprintf(m, sizeof(m),
                     "destination block client=%016" PRIx64 " base=%#" PRIx64
                     " is not registered",
                     e.dst_client, e.dst_base);
            return fail(ProxyStatus::kUnknownDestination, i, m);
        }
        const bool refused = src->device != dst->device &&
                             g_policy.denied(src->device, dst->device);
        if (refused && !hostStageable(src->device, dst->device)) {
            // Before any merging or mapping: the engine moves the request
            // to its base transport.
            const int sd = src->device, dd = dst->device;
            char m[192];
            snprintf(m, sizeof(m),
                     "P2P from gpu %d (%s) to gpu %d (%s) is denied: %s", sd,
                     g_devices[sd].pci.c_str(), dd, g_devices[dd].pci.c_str(),
                     PeerPolicy::reasonName(g_policy.reason(sd, dd)));
            r.denied = true;
            return fail(ProxyStatus::kNoPeerAccess, i, m);
        }
        if (e.src_offset > src->size || e.length > src->size - e.src_offset ||
            e.dst_offset > dst->size || e.length > dst->size - e.dst_offset) {
            return fail(ProxyStatus::kOutOfRange, i,
                        "copy range exceeds a registered block");
        }
        if (g_cfg.coalesce && !items.empty()) {
            Item &p = items.back();
            if (p.src == src && p.dst == dst &&
                p.src_off + p.len == e.src_offset &&
                p.dst_off + p.len == e.dst_offset) {
                p.len += e.length;
                r.bytes += e.length;
                continue;
            }
        }
        items.push_back(
            {src, dst, e.src_offset, e.dst_offset, e.length, refused});
        r.bytes += e.length;
    }

    // 2. Choose the executing GPU, map both blocks in its context and split
    //    into kernel (small) and copy-engine (large) work.
    struct DevWork {
        std::vector<void *> ce_dst, ce_src;
        std::vector<size_t> ce_len;
        std::vector<GatherEntry> small;
        uint64_t ce_bytes = 0, small_bytes = 0;
    };
    std::map<int, DevWork> work;
    // Refused pairs, host-staged: (sending GPU, receiving GPU) -> items,
    // each block mapped in its own GPU's context (no peer access needed).
    std::map<std::pair<int, int>, std::vector<StagedItem>> staged;
    uint64_t staged_bytes = 0;
    MapResolver maps;
    // Maps both blocks in their own GPU's context and queues |it| for
    // host staging; the CUDA error otherwise.
    auto stage = [&](const Item &it) -> cudaError_t {
        char *src_map = nullptr, *dst_map = nullptr;
        cudaError_t ce = maps.map(it.src, it.src->device, &src_map);
        if (ce == cudaSuccess) ce = maps.map(it.dst, it.dst->device, &dst_map);
        if (ce != cudaSuccess) return ce;
        staged[{it.src->device, it.dst->device}].push_back(
            {src_map + it.src_off, dst_map + it.dst_off, it.len});
        staged_bytes += it.len;
        return cudaSuccess;
    };
    for (size_t k = 0; k < items.size(); ++k) {
        const Item &it = items[k];
        if (it.staged) {
            cudaError_t ce = stage(it);
            if (ce != cudaSuccess) {
                if (isStickyCudaError(ce)) dieOnStickyError(ce, "IPC open");
                cudaGetLastError();
                return fail(ProxyStatus::kCudaError, count,
                            std::string("cudaIpcOpenMemHandle: ") +
                                cudaGetErrorString(ce));
            }
            continue;
        }
        const bool small = it.len < g_cfg.gather_threshold;
        int dev = it.dst->device;
        if (small && g_cfg.gather_on_src) dev = it.src->device;
        if (small && g_kernel_unusable[dev].load(std::memory_order_relaxed))
            dev = it.dst->device;  // copy engines of the receiving GPU
        const int other =
            (dev == it.dst->device) ? it.src->device : it.dst->device;
        if (other != dev && !g_peer[dev][other].load()) {
            // No P2P between them at all: host-staged when verified.
            if (hostStageable(it.src->device, it.dst->device)) {
                cudaError_t ce = stage(it);
                if (ce == cudaSuccess) continue;
                if (isStickyCudaError(ce)) dieOnStickyError(ce, "IPC open");
                cudaGetLastError();
                return fail(ProxyStatus::kCudaError, count,
                            std::string("cudaIpcOpenMemHandle: ") +
                                cudaGetErrorString(ce));
            }
            char m[96];
            snprintf(m, sizeof(m), "gpu %d cannot access gpu %d (no P2P)", dev,
                     other);
            return fail(ProxyStatus::kNoPeerAccess, count, m);
        }
        char *src_map = nullptr, *dst_map = nullptr;
        cudaError_t ce = maps.map(it.src, dev, &src_map);
        if (ce == cudaSuccess) ce = maps.map(it.dst, dev, &dst_map);
        if (ce != cudaSuccess) {
            if (isStickyCudaError(ce)) dieOnStickyError(ce, "IPC open");
            cudaGetLastError();
            return fail(
                ProxyStatus::kCudaError, count,
                std::string("cudaIpcOpenMemHandle: ") + cudaGetErrorString(ce));
        }
        DevWork &w = work[dev];
        char *d = dst_map + it.dst_off;
        char *s = src_map + it.src_off;
        if (small && !g_kernel_unusable[dev].load(std::memory_order_relaxed)) {
            w.small.push_back({reinterpret_cast<uint64_t>(d),
                               reinterpret_cast<uint64_t>(s), it.len});
            w.small_bytes += it.len;
        } else {
            w.ce_dst.push_back(d);
            w.ce_src.push_back(s);
            w.ce_len.push_back(it.len);
            w.ce_bytes += it.len;
        }
    }
    const uint64_t t_planned = nowUs();
    g_stats.plan_us_total += t_planned - t_recv_us;
    g_stats.coalesced_entries += count - items.size();

    if (budget_ms && nowUs() - t_recv_us > uint64_t(budget_ms) * 1000) {
        return fail(ProxyStatus::kDeadlineExceeded, count,
                    "start budget exceeded before the copy was issued");
    }

    // The engine gives up on a copy after twice its start budget (its
    // request timeout; budget = timeout / 2) and rewrites it on its base
    // transport. Nothing of this copy may be written after that: no staged
    // piece (at most one slot, a few ms) is issued past |hard_deadline| -- a
    // tenth of the budget (at least 50 ms) before the engine gives up -- and
    // whatever is in flight is drained before the refusal goes out.
    const uint64_t margin_us =
        std::max<uint64_t>(uint64_t(budget_ms) * 100, 50000);
    const uint64_t hard_deadline =
        budget_ms
            ? t_recv_us + uint64_t(budget_ms) * 2000 -
                  std::min<uint64_t>(margin_us, uint64_t(budget_ms) * 1000)
            : UINT64_MAX;
    auto started_too_late = [&]() {
        const uint64_t now = nowUs();
        return (budget_ms && now - t_recv_us > uint64_t(budget_ms) * 1000) ||
               now >= hard_deadline;
    };

    // 3a. Host-staged copies first (they leave both GPUs' streams idle). A
    //     copy waits at most --host-staging-wait-ms for staging slots; on any
    //     failure the request is refused like a denied pair, so the engine
    //     moves it to its base transport (which rewrites every byte).
    if (!staged.empty()) {
        for (const auto &kv : staged) {
            const int sd = kv.first.first, dd = kv.first.second;
            std::string why;
            bool ok = false;
            if (started_too_late()) {
                why = "start budget exceeded";
            } else {
                const uint64_t wait_until =
                    nowUs() + uint64_t(g_host_staging_wait_ms) * 1000;
                ok = stagedCopy(sd, dd, kv.second, wait_until, hard_deadline,
                                &why);
            }
            if (!ok) {
                g_stats.staged_failures++;
                static std::atomic<uint64_t> last_log_us{0};
                uint64_t last = last_log_us.load();
                const uint64_t now = nowUs();
                if (now - last >= 5000000 &&
                    last_log_us.compare_exchange_strong(last, now))
                    LOGW(
                        "host-staged copy gpu %d (%s) -> gpu %d (%s) not "
                        "done: %s; refused (base transport)",
                        sd, g_devices[sd].pci.c_str(), dd,
                        g_devices[dd].pci.c_str(), why.c_str());
                char m[256];
                snprintf(
                    m, sizeof(m),
                    "copy from gpu %d (%s) to gpu %d (%s) not host-staged: "
                    "%s",
                    sd, g_devices[sd].pci.c_str(), dd,
                    g_devices[dd].pci.c_str(), why.c_str());
                r.denied = true;
                return fail(ProxyStatus::kNoPeerAccess, count, m);
            }
        }
        g_stats.staged_copies++;
        g_stats.staged_bytes += staged_bytes;
        if (!work.empty() && started_too_late())
            return fail(ProxyStatus::kDeadlineExceeded, count,
                        "start budget exceeded before the P2P part was issued");
    }

    // 3. Issue: one kernel launch (plus its table upload) and one batched
    //    copy-engine submission per executing GPU, then wait for all.
    std::vector<cudaStream_t> used;
    auto drain = [&]() {
        for (auto s : used) cudaStreamSynchronize(s);
    };
    for (auto &kv : work) {
        const int dev = kv.first;
        DevWork &w = kv.second;
        DeviceResources *res = nullptr;
        cudaError_t ce = t_res.get(dev, &res);
        if (ce == cudaSuccess) ce = cudaSetDevice(dev);
        if (ce == cudaSuccess && !w.small.empty()) {
            ce = t_res.reserve(dev, *res, w.small.size());
            if (ce == cudaSuccess) {
                memcpy(res->host_table, w.small.data(),
                       w.small.size() * sizeof(GatherEntry));
                ce = cudaMemcpyAsync(res->dev_table, res->host_table,
                                     w.small.size() * sizeof(GatherEntry),
                                     cudaMemcpyHostToDevice, res->stream);
            }
            if (ce == cudaSuccess) {
                ce = launchGatherCopy(res->dev_table,
                                      static_cast<uint32_t>(w.small.size()),
                                      res->grid_blocks, res->stream);
                if (ce == cudaErrorNoKernelImageForDevice ||
                    ce == cudaErrorInvalidDeviceFunction ||
                    ce == cudaErrorUnsupportedPtxVersion) {
                    // No usable kernel image for this GPU: use the copy
                    // engines for this and all later batches.
                    cudaGetLastError();
                    g_kernel_unusable[dev] = true;
                    LOGW(
                        "gather kernel unavailable on gpu %d (%s); small "
                        "copies use the copy engines",
                        dev, cudaGetErrorString(ce));
                    for (auto &g : w.small) {
                        w.ce_dst.push_back(reinterpret_cast<void *>(g.dst));
                        w.ce_src.push_back(reinterpret_cast<void *>(g.src));
                        w.ce_len.push_back(g.length);
                    }
                    w.ce_bytes += w.small_bytes;
                    w.small_bytes = 0;
                    w.small.clear();
                    ce = cudaSuccess;
                }
            }
        }
        if (ce == cudaSuccess)
            ce = issueCopies(w.ce_dst, w.ce_src, w.ce_len, res->stream);
        if (res && res->stream) used.push_back(res->stream);
        if (ce != cudaSuccess) {
            if (isStickyCudaError(ce)) dieOnStickyError(ce, "copy submit");
            cudaGetLastError();
            drain();  // nothing may still be running when we reply
            return fail(ProxyStatus::kCudaError, count,
                        std::string("copy submit: ") + cudaGetErrorString(ce));
        }
        g_stats.kernel_entries += w.small.size();
        g_stats.kernel_bytes += w.small_bytes;
        g_stats.ce_entries += w.ce_len.size();
        g_stats.ce_bytes += w.ce_bytes;
    }
    cudaError_t first_err = cudaSuccess;
    for (auto s : used) {
        cudaError_t ce = cudaStreamSynchronize(s);
        if (ce != cudaSuccess && first_err == cudaSuccess) first_err = ce;
    }
    g_stats.exec_us_total += nowUs() - t_planned;
    if (first_err != cudaSuccess) {
        if (isStickyCudaError(first_err))
            dieOnStickyError(first_err, "copy completion");
        cudaGetLastError();
        return fail(ProxyStatus::kCudaError, count,
                    std::string("copy: ") + cudaGetErrorString(first_err));
    }
    return r;
}

// ------------------------------------------------------------- connections
std::atomic<uint64_t> g_next_conn_id{1};

struct Connection {
    int fd;
    uint64_t conn_id = g_next_conn_id++;
    bool hello = false;
    bool control = false;
    uint64_t client_id = 0;
    uint32_t pid = 0;
    std::string node;
    // Registrations created through this (control) connection.
    std::map<uint64_t, std::shared_ptr<Registration>> owned;
    // GPUs this connection registered blocks on (self-test trigger).
    std::set<int> devices;
};

int findDevice(const uint8_t *uuid) {
    for (auto &d : g_devices)
        if (memcmp(d.uuid, uuid, kUuidSize) == 0) return d.ordinal;
    return -1;
}

void replyError(int fd, MsgType type, uint64_t seq, ProxyStatus st,
                const std::string &msg) {
    sendMsg(fd, type, seq, st, msg.data(), msg.size());
}

void dropOwned(Connection &c) {
    if (c.owned.empty()) return;
    size_t dropped = 0;
    {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        for (auto &kv : c.owned) {
            auto it = g_regs.find({c.client_id, kv.first});
            // A reconnected client may already have re-registered the key;
            // only drop registrations this connection still owns.
            if (it != g_regs.end() && it->second == kv.second &&
                it->second->owner_conn == c.conn_id) {
                g_regs.erase(it);
                ++dropped;
            }
        }
    }
    g_stats.registrations -= dropped;
    // Mappings are closed when the last in-flight copy releases them.
    c.owned.clear();
    LOGI("client %016" PRIx64
         " (pid %u, node %s) disconnected; released %zu "
         "registrations",
         c.client_id, c.pid, c.node.c_str(), dropped);
}

void handleConnection(int fd) {
    g_stats.connections++;
    g_stats.active_connections++;
    Connection c;
    c.fd = fd;
    std::vector<char> payload;
    while (true) {
        MsgHeader h;
        if (!readAll(fd, &h, sizeof(h))) break;
        if (h.magic != kMagic) {
            LOGW("bad magic on fd %d, closing", fd);
            break;
        }
        if (h.payload_len > kMaxPayload) {
            LOGW("oversized payload (%u) on fd %d, closing", h.payload_len, fd);
            break;
        }
        payload.resize(h.payload_len);
        if (h.payload_len && !readAll(fd, payload.data(), h.payload_len)) break;
        const uint64_t t_recv = nowUs();
        const auto type = static_cast<MsgType>(h.type);
        if (h.version != kProtocolVersion) {
            replyError(fd, type, h.seq, ProxyStatus::kVersionMismatch,
                       "daemon speaks protocol version " +
                           std::to_string(kProtocolVersion));
            continue;
        }
        bool ok = true;
        switch (type) {
            case MsgType::kHello: {
                if (payload.size() != sizeof(HelloReq)) {
                    replyError(fd, type, h.seq, ProxyStatus::kBadRequest,
                               "bad HELLO");
                    break;
                }
                HelloReq req;
                memcpy(&req, payload.data(), sizeof(req));
                if (req.client_id == 0 ||
                    (c.hello && c.client_id != req.client_id)) {
                    replyError(fd, type, h.seq, ProxyStatus::kBadRequest,
                               "invalid client id");
                    break;
                }
                c.client_id = req.client_id;
                c.pid = req.pid;
                c.node.assign(req.node_id, strnlen(req.node_id, kNodeIdSize));
                bool was_control = c.control;
                c.control = c.control || (req.flags & kHelloFlagControl);
                c.hello = true;
                if (c.control && !was_control) {
                    g_stats.active_clients++;
                    LOGI("client %016" PRIx64 " (pid %u, node %s) connected",
                         c.client_id, c.pid, c.node.c_str());
                }
                HelloResp resp{};
                resp.daemon_epoch = g_epoch;
                resp.num_devices = static_cast<uint32_t>(g_devices.size());
                ok = sendMsg(fd, type, h.seq, ProxyStatus::kOk, &resp,
                             sizeof(resp));
                break;
            }
            case MsgType::kRegister: {
                if (!c.control) {
                    replyError(fd, type, h.seq,
                               ProxyStatus::kNotControlConnection,
                               "REGISTER requires a control connection");
                    break;
                }
                if (payload.size() != sizeof(RegisterReq)) {
                    replyError(fd, type, h.seq, ProxyStatus::kBadRequest,
                               "bad REGISTER");
                    break;
                }
                RegisterReq req;
                memcpy(&req, payload.data(), sizeof(req));
                int dev = findDevice(req.gpu_uuid);
                if (dev < 0) {
                    replyError(fd, type, h.seq, ProxyStatus::kUnknownDevice,
                               hexUuid(req.gpu_uuid) +
                                   " is not visible to the daemon");
                    break;
                }
                if (!ensureDevice(dev)) {
                    replyError(fd, type, h.seq, ProxyStatus::kCudaError,
                               "daemon cannot create a context on " +
                                   hexUuid(req.gpu_uuid) +
                                   " (see daemon log; exclusive compute "
                                   "mode is not supported)");
                    break;
                }
                if (req.size == 0 || req.base + req.size < req.base) {
                    replyError(fd, type, h.seq, ProxyStatus::kBadRequest,
                               "bad block range");
                    break;
                }
                // An engine's first block on this GPU: test the GPU's pairs
                // that have not passed yet (in the background; until then,
                // in enforce mode, copies between them are refused), before
                // the block is visible to any copy.
                if (c.devices.insert(dev).second) requestSelfTestForDevice(dev);
                bool replaced = false, same = false;
                {
                    std::lock_guard<std::mutex> lock(g_reg_mu);
                    auto it = g_regs.find({c.client_id, req.base});
                    if (it != g_regs.end()) {
                        auto &old = it->second;
                        same = old->size == req.size && old->device == dev &&
                               memcmp(old->handle, req.ipc_handle,
                                      kIpcHandleSize) == 0;
                        replaced = !same;
                    }
                    if (!same) {
                        auto reg = std::make_shared<Registration>();
                        reg->client_id = c.client_id;
                        reg->base = req.base;
                        reg->size = req.size;
                        reg->device = dev;
                        memcpy(reg->handle, req.ipc_handle, kIpcHandleSize);
                        reg->owner_conn = c.conn_id;
                        g_regs[{c.client_id, req.base}] = reg;
                        c.owned[req.base] = reg;
                        if (!replaced) g_stats.registrations++;
                    } else {
                        it->second->owner_conn = c.conn_id;
                        c.owned[req.base] = it->second;
                    }
                }
                LOGI("register client=%016" PRIx64 " base=%#" PRIx64
                     " size=%" PRIu64 " gpu=%d (%s)%s",
                     c.client_id, req.base, req.size, dev,
                     g_devices[dev].pci.c_str(),
                     same ? " [unchanged]" : (replaced ? " [replaced]" : ""));
                ok = sendMsg(fd, type, h.seq, ProxyStatus::kOk, nullptr, 0);
                break;
            }
            case MsgType::kUnregister: {
                if (!c.control || payload.size() != sizeof(UnregisterReq)) {
                    replyError(fd, type, h.seq, ProxyStatus::kBadRequest,
                               "bad UNREGISTER");
                    break;
                }
                UnregisterReq req;
                memcpy(&req, payload.data(), sizeof(req));
                size_t erased = 0;
                {
                    std::lock_guard<std::mutex> lock(g_reg_mu);
                    erased = g_regs.erase({c.client_id, req.base});
                }
                c.owned.erase(req.base);
                g_stats.registrations -= erased;
                LOGI("unregister client=%016" PRIx64 " base=%#" PRIx64 "%s",
                     c.client_id, req.base, erased ? "" : " [unknown]");
                ok = sendMsg(fd, type, h.seq, ProxyStatus::kOk, nullptr, 0);
                break;
            }
            case MsgType::kCopy: {
                g_inflight_copies++;
                if (g_stop) {
                    // Shutting down: close the connection without starting
                    // the copy; the client retries with the next daemon.
                    g_inflight_copies--;
                    ok = false;
                    break;
                }
                struct InflightGuard {
                    ~InflightGuard() { g_inflight_copies--; }
                } inflight_guard;
                CopyReqHeader rh;
                if (payload.size() < sizeof(rh)) {
                    replyError(fd, type, h.seq, ProxyStatus::kBadRequest,
                               "bad COPY");
                    break;
                }
                memcpy(&rh, payload.data(), sizeof(rh));
                if (rh.count > kMaxCopyEntries ||
                    payload.size() !=
                        sizeof(rh) + size_t(rh.count) * sizeof(CopyEntry)) {
                    replyError(fd, type, h.seq, ProxyStatus::kBadRequest,
                               "bad COPY size");
                    break;
                }
                // payload is char-aligned; copy entries out for alignment.
                std::vector<CopyEntry> entries(rh.count);
                if (rh.count)
                    memcpy(entries.data(), payload.data() + sizeof(rh),
                           size_t(rh.count) * sizeof(CopyEntry));
                CopyResult res =
                    doCopy(entries.data(), rh.count, rh.budget_ms, t_recv);
                uint64_t us = nowUs() - t_recv;
                g_stats.copy_requests++;
                g_stats.copy_entries += rh.count;
                g_stats.copy_us_total += us;
                updateMax(g_stats.copy_us_max, us);
                CopyResp resp{};
                resp.failed_index = res.failed_index;
                resp.copied_bytes =
                    res.status == ProxyStatus::kOk ? res.bytes : 0;
                std::string out(reinterpret_cast<char *>(&resp), sizeof(resp));
                if (res.status == ProxyStatus::kOk) {
                    g_stats.copy_bytes += res.bytes;
                } else if (res.denied) {
                    // Expected for every copy between the GPUs of a denied
                    // pair: counted apart from failures, logged rarely.
                    g_stats.denied_copies++;
                    out += res.message;
                    static std::atomic<uint64_t> last_denied_log_us{0};
                    static std::atomic<uint64_t> denied_since_log{0};
                    const uint64_t now_us = nowUs();
                    uint64_t last = last_denied_log_us.load();
                    denied_since_log++;
                    if ((last == 0 || now_us - last >= 60000000) &&
                        last_denied_log_us.compare_exchange_strong(last,
                                                                   now_us)) {
                        LOGI("refused %" PRIu64
                             " copy request(s) from client %016" PRIx64
                             " and others since the last report: %s; the "
                             "engines use their base transport for them",
                             denied_since_log.exchange(0), c.client_id,
                             res.message.c_str());
                    }
                } else {
                    g_stats.copy_failures++;
                    out += res.message;
                    // At most one line per second; a restart can make
                    // clients retry many batches against peers that have
                    // not re-registered yet.
                    static std::atomic<uint64_t> last_log_us{0};
                    static std::atomic<uint64_t> suppressed{0};
                    const uint64_t now_us = nowUs();
                    uint64_t last = last_log_us.load();
                    if (now_us - last >= 1000000 &&
                        last_log_us.compare_exchange_strong(last, now_us)) {
                        const uint64_t more = suppressed.exchange(0);
                        const std::string tail =
                            more ? " [+" + std::to_string(more) + " similar]"
                                 : std::string();
                        LOGW("copy from client %016" PRIx64
                             " failed: %s (%s, entry %u of %u)%s",
                             c.client_id, proxyStatusName(res.status),
                             res.message.c_str(), res.failed_index, rh.count,
                             tail.c_str());
                    } else {
                        suppressed++;
                    }
                }
                ok = sendMsg(fd, type, h.seq, res.status, out.data(),
                             out.size());
                break;
            }
            case MsgType::kPing: {
                if (g_stop) {
                    // Shutting down: report the daemon as gone so clients
                    // hold their requests for the next instance.
                    ok = false;
                    break;
                }
                PingResp resp{};
                resp.daemon_epoch = g_epoch;
                resp.uptime_ms = (nowUs() - g_start_us) / 1000;
                ok = sendMsg(fd, type, h.seq, ProxyStatus::kOk, &resp,
                             sizeof(resp));
                break;
            }
            case MsgType::kStats: {
                std::string s = statsText();
                ok = sendMsg(fd, type, h.seq, ProxyStatus::kOk, s.data(),
                             s.size());
                break;
            }
            default:
                replyError(fd, type, h.seq, ProxyStatus::kBadRequest,
                           "unknown message type");
        }
        if (!ok) break;
    }
    if (c.control) {
        dropOwned(c);
        g_stats.active_clients--;
    }
    close(fd);
    g_stats.active_connections--;
}

// ------------------------------------------------------------------- setup
bool initDevices(bool quiet = false) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        LOGE(
            "cudaGetDeviceCount: %s (is the daemon container able to see "
            "the GPUs? e.g. NVIDIA_VISIBLE_DEVICES=all)",
            cudaGetErrorString(e));
        return false;
    }
    if (n > kMaxDevices) {
        LOGW("%d gpus visible, using the first %d", n, kMaxDevices);
        n = kMaxDevices;
    }
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp prop;
        e = cudaGetDeviceProperties(&prop, i);
        if (e != cudaSuccess) {
            LOGE("cudaGetDeviceProperties(%d): %s", i, cudaGetErrorString(e));
            return false;
        }
        Device d;
        d.ordinal = i;
        memcpy(d.uuid, prop.uuid.bytes, kUuidSize);
        d.name = prop.name;
        char pci[32];
        snprintf(pci, sizeof(pci), "%04x:%02x:%02x", prop.pciDomainID,
                 prop.pciBusID, prop.pciDeviceID);
        d.pci = pci;
        d.addr.domain = static_cast<uint32_t>(prop.pciDomainID);
        d.addr.bus = static_cast<uint32_t>(prop.pciBusID);
        d.addr.device = static_cast<uint32_t>(prop.pciDeviceID);
        d.numa = readGpuNumaNode(d.addr);
        g_devices.push_back(d);
        g_pci_short.push_back(formatPciAddrShort(d.addr));
        // cudaDevAttrComputeMode: cudaDeviceProp lost the field in CUDA 13.
        int compute_mode = cudaComputeModeDefault;
        cudaDeviceGetAttribute(&compute_mode, cudaDevAttrComputeMode, i);
        const bool exclusive = compute_mode != cudaComputeModeDefault;
        g_devices.back().exclusive = exclusive;
        if (!quiet)
            LOGI("gpu %d: %s %s pci %s%s", i, d.name.c_str(),
                 hexUuid(d.uuid).c_str(), d.pci.c_str(),
                 exclusive ? " compute mode NOT DEFAULT" : "");
        if (exclusive && !quiet) {
            LOGW(
                "gpu %d is in an exclusive/prohibited compute mode; the "
                "daemon cannot create a context next to an engine using it, "
                "so copies involving it will be refused (engines fall back "
                "to their base transport). The proxy needs compute mode "
                "DEFAULT.",
                i);
        }
    }
    g_policy.reset(n);
    for (int a = 0; a < kMaxDevices; ++a) {
        g_dev_ready[a] = false;
        g_kernel_unusable[a] = false;
        g_host_state[a] = kNotTested;
        for (int b = 0; b < kMaxDevices; ++b) {
            g_peer[a][b] = false;
            g_test_state[a][b] = kNotTested;
            g_stage_state[a][b] = kNotTested;
        }
    }
    return n > 0;
}

// --------------------------------------------- P2P self-test (child side)
// Runs in a child process (--p2p-selftest-child), never in the daemon
// itself: a hang or an unrecoverable CUDA error there cannot take the daemon
// down. For each requested pair it copies patterns in both directions with
// the production code paths (issueCopies on the receiving GPU's stream, the
// gather kernel on the GPU --gather-on picks, this thread's
// DeviceResources) and reads the whole destination buffer back: copied
// ranges must hold the source GPU's bytes, everything else the poison the
// buffer was filled with just before (a dropped or stray write shows). Each
// GPU gets a source and a destination buffer of up to 4 MiB (or what the
// gather threshold needs; smaller sizes down to that are tried when memory
// is short). Only a data mismatch fails a direction; anything that keeps a
// path from running leaves it untested.
constexpr int kSelfTestIterations = 3;
bool enablePeer(int dst, int src);

struct PathResult {
    bool ran = false;       // compared at least once
    bool corrupt = false;   // read back wrong data
    bool untested = false;  // could not run (memory, CUDA error, ...)
    std::string detail;
};

struct DirectionResult {
    int src = -1, dst = -1;
    bool ce_peer = false;      // the receiving GPU can reach the sender
    bool kernel_peer = false;  // the kernel's GPU can reach the other one
    uint64_t bytes = 0;        // test buffer size used
    std::string untested_reason;
    PathResult ce, kernel;
    int kernel_dev = -1;
    uint64_t ce_bytes = 0, ce_us = 0;

    DirVerdict verdict() const {
        if (!ce_peer && !kernel_peer) return DirVerdict::kNoPeer;
        if (ce.corrupt || kernel.corrupt) return DirVerdict::kFail;
        if (!untested_reason.empty() || ce.untested || kernel.untested ||
            (ce_peer && !ce.ran) || (!ce.ran && !kernel.ran))
            return DirVerdict::kUntested;
        return DirVerdict::kPass;
    }
};

struct TestBuffers {
    char *src = nullptr;  // holds this GPU's pattern
    char *dst = nullptr;  // receives copies from other GPUs
    uint64_t bytes = 0;
    std::string error;
};

bool allocTestBuffers(int dev, TestBuffers &b) {
    const uint64_t min_bytes = selfTestMinBufferBytes(g_cfg.gather_threshold);
    std::vector<uint64_t> sizes;
    for (uint64_t size = std::max<uint64_t>(4ull << 20, min_bytes);
         size > min_bytes; size /= 2)
        sizes.push_back(size);
    sizes.push_back(min_bytes);
    for (uint64_t size : sizes) {
        cudaError_t e = cudaSetDevice(dev);
        if (e == cudaSuccess)
            e = cudaMalloc(reinterpret_cast<void **>(&b.src), size);
        if (e == cudaSuccess) {
            e = cudaMalloc(reinterpret_cast<void **>(&b.dst), size);
            if (e != cudaSuccess) {
                cudaGetLastError();
                cudaFree(b.src);
            }
        }
        if (e == cudaSuccess) {
            b.bytes = size;
            return true;
        }
        if (isStickyCudaError(e)) dieOnStickyError(e, "P2P self-test alloc");
        cudaGetLastError();
        b.src = b.dst = nullptr;
        b.error = std::string("cudaMalloc on gpu ") + std::to_string(dev) +
                  ": " + cudaGetErrorString(e);
    }
    return false;
}

void freeTestBuffers(int dev, TestBuffers &b) {
    if (!b.bytes) return;
    cudaSetDevice(dev);
    cudaFree(b.src);
    cudaFree(b.dst);
    cudaGetLastError();
    b = TestBuffers();
}

// Upload |n| bytes to device memory and wait until the DMA has landed (a
// pageable cudaMemcpy may return before it has).
cudaError_t uploadSync(int dev, void *dst, const void *src, uint64_t n) {
    cudaError_t e = cudaSetDevice(dev);
    if (e == cudaSuccess) e = cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaStreamSynchronize(cudaStreamLegacy);
    return e;
}

// One pass: |plan| from |s| to |d| on |exec|'s stream through the gather
// kernel or the copy engines, waited for. *unusable: the kernel has no image
// for |exec| (production then uses the copy engines there).
cudaError_t runSelfTestPass(int exec, bool kernel, char *s, char *d,
                            const std::vector<TestCopy> &plan,
                            uint64_t *elapsed_us, bool *unusable) {
    *unusable = false;
    DeviceResources *res = nullptr;
    cudaError_t e = t_res.get(exec, &res);
    if (e == cudaSuccess) e = cudaSetDevice(exec);
    if (e != cudaSuccess) return e;
    const uint64_t t0 = nowUs();
    if (kernel) {
        std::vector<GatherEntry> table;
        table.reserve(plan.size());
        for (const auto &c : plan)
            table.push_back({reinterpret_cast<uint64_t>(d + c.dst_off),
                             reinterpret_cast<uint64_t>(s + c.src_off), c.len});
        e = t_res.reserve(exec, *res, table.size());
        if (e != cudaSuccess) return e;
        memcpy(res->host_table, table.data(),
               table.size() * sizeof(GatherEntry));
        e = cudaMemcpyAsync(res->dev_table, res->host_table,
                            table.size() * sizeof(GatherEntry),
                            cudaMemcpyHostToDevice, res->stream);
        if (e == cudaSuccess)
            e = launchGatherCopy(res->dev_table,
                                 static_cast<uint32_t>(table.size()),
                                 res->grid_blocks, res->stream);
        if (e == cudaErrorNoKernelImageForDevice ||
            e == cudaErrorInvalidDeviceFunction ||
            e == cudaErrorUnsupportedPtxVersion) {
            cudaGetLastError();
            cudaStreamSynchronize(res->stream);
            *unusable = true;
            return cudaSuccess;
        }
    } else {
        std::vector<void *> dsts, srcs;
        std::vector<size_t> sizes;
        for (const auto &c : plan) {
            dsts.push_back(d + c.dst_off);
            srcs.push_back(s + c.src_off);
            sizes.push_back(c.len);
        }
        e = issueCopies(dsts, srcs, sizes, res->stream);
    }
    cudaError_t se = cudaStreamSynchronize(res->stream);
    if (e == cudaSuccess) e = se;
    *elapsed_us = nowUs() - t0;
    return e;
}

// Tests the ordered pairs |dirs| (peer access enabled where possible):
// one result per direction, in order.
std::vector<DirectionResult> runSelfTests(
    const std::vector<std::pair<int, int>> &dirs) {
    std::vector<DirectionResult> results(dirs.size());
    std::map<int, TestBuffers> bufs;
    const bool kernel_wanted = g_cfg.gather_threshold > 0;
    for (size_t i = 0; i < dirs.size(); ++i) {
        DirectionResult &r = results[i];
        r.src = dirs[i].first;
        r.dst = dirs[i].second;
        const int kexec = g_cfg.gather_on_src ? r.src : r.dst;
        const int kother = kexec == r.src ? r.dst : r.src;
        r.ce_peer = g_peer[r.dst][r.src].load();
        r.kernel_peer = kernel_wanted && g_peer[kexec][kother].load();
        if (!r.ce_peer && !r.kernel_peer) continue;  // not judged
        for (int dev : {r.src, r.dst})
            if (!bufs.count(dev)) allocTestBuffers(dev, bufs[dev]);
    }
    uint64_t max_bytes = 0;
    for (DirectionResult &r : results) {
        if (!r.ce_peer && !r.kernel_peer) continue;
        const TestBuffers &bs = bufs[r.src], &bd = bufs[r.dst];
        if (!bs.bytes || !bd.bytes) {
            r.untested_reason = !bs.bytes ? bs.error : bd.error;
            continue;
        }
        r.bytes = std::min(bs.bytes, bd.bytes);
        max_bytes = std::max(max_bytes, r.bytes);
    }
    std::map<int, std::vector<uint8_t>> src_img;  // per source GPU
    for (const DirectionResult &r : results)
        if (r.bytes) src_img[r.src].resize(bufs[r.src].bytes);
    std::vector<uint8_t> poison(max_bytes), expect(max_bytes), got(max_bytes);
    std::set<int> upload_failed;
    auto untested = [](PathResult &p, std::string why) {
        p.untested = true;
        p.detail = std::move(why);
    };
    for (int iter = 0; iter < kSelfTestIterations && max_bytes; ++iter) {
        // A different pattern per source GPU and iteration: a copy that
        // read the wrong GPU or a stale buffer does not match.
        for (auto &kv : src_img) {
            fillPattern(kv.second.data(), kv.second.size(),
                        mixSeed(g_epoch ^ (uint64_t(kv.first) << 48), iter));
            cudaError_t e = uploadSync(kv.first, bufs[kv.first].src,
                                       kv.second.data(), kv.second.size());
            if (e != cudaSuccess) {
                if (isStickyCudaError(e)) dieOnStickyError(e, "P2P self-test");
                cudaGetLastError();
                upload_failed.insert(kv.first);
            }
        }
        for (int pass = 0; pass < 2; ++pass) {
            const bool kernel = pass == 1;
            fillPattern(poison.data(), max_bytes,
                        ~mixSeed(g_epoch, uint64_t(iter) * 2 + pass));
            for (DirectionResult &r : results) {
                PathResult &pr = kernel ? r.kernel : r.ce;
                if (!r.bytes || pr.corrupt || pr.untested) continue;
                if (!(kernel ? r.kernel_peer : r.ce_peer)) continue;
                const int exec =
                    kernel ? (g_cfg.gather_on_src ? r.src : r.dst) : r.dst;
                if (kernel && g_kernel_unusable[exec].load()) continue;
                if (upload_failed.count(r.src)) {
                    untested(pr, "pattern upload failed");
                    continue;
                }
                const std::vector<TestCopy> plan =
                    selfTestPlan(r.bytes, static_cast<uint32_t>(iter), !kernel,
                                 g_cfg.gather_threshold);
                if (plan.empty()) continue;  // kernel: nothing below threshold
                uint64_t us = 0;
                bool unusable = false;
                cudaError_t e =
                    uploadSync(r.dst, bufs[r.dst].dst, poison.data(), r.bytes);
                if (e == cudaSuccess)
                    e = runSelfTestPass(exec, kernel, bufs[r.src].src,
                                        bufs[r.dst].dst, plan, &us, &unusable);
                if (e == cudaSuccess && unusable) {
                    g_kernel_unusable[exec] = true;
                    LOGW(
                        "gather kernel unavailable on gpu %d; small copies "
                        "use the copy engines",
                        exec);
                    continue;
                }
                if (e == cudaSuccess) {
                    e = cudaSetDevice(r.dst);
                    if (e == cudaSuccess)
                        e = cudaMemcpy(got.data(), bufs[r.dst].dst, r.bytes,
                                       cudaMemcpyDeviceToHost);
                }
                if (e != cudaSuccess) {
                    if (isStickyCudaError(e))
                        dieOnStickyError(e, "P2P self-test");
                    cudaGetLastError();
                    untested(pr, std::string("CUDA error: ") +
                                     cudaGetErrorString(e));
                    continue;
                }
                pr.ran = true;
                if (kernel) r.kernel_dev = exec;
                expectedImage(src_img[r.src].data(), poison.data(), r.bytes,
                              plan, expect.data());
                const Mismatch m =
                    compareImages(expect.data(), got.data(), r.bytes);
                if (m.any()) {
                    uint64_t copied = 0;
                    for (const auto &c : plan) copied += c.len;
                    char why[224];
                    snprintf(why, sizeof(why),
                             "CORRUPT (iteration %d: %" PRIu64 " of %" PRIu64
                             " bytes wrong, %" PRIu64
                             " copied; first at +%#" PRIx64
                             ": expected %#04x, read %#04x)",
                             iter, m.bytes, r.bytes, copied, m.first,
                             m.expected, m.actual);
                    pr.corrupt = true;
                    pr.detail = why;
                } else if (!kernel) {
                    for (const auto &c : plan) r.ce_bytes += c.len;
                    r.ce_us += us;
                }
            }
        }
    }
    for (auto &kv : bufs) freeTestBuffers(kv.first, kv.second);
    return results;
}

std::string describePath(const char *name, const PathResult &p) {
    if (p.corrupt) return std::string(name) + " " + p.detail;
    if (p.untested) return std::string(name) + " untested (" + p.detail + ")";
    if (!p.ran) return std::string(name) + " not tested";
    return std::string(name) + " OK";
}

std::string describeDirection(const DirectionResult &r) {
    char head[160];
    snprintf(head, sizeof(head), "gpu %d (%s) -> gpu %d (%s): ", r.src,
             g_devices[r.src].pci.c_str(), r.dst, g_devices[r.dst].pci.c_str());
    std::string out = head;
    if (r.verdict() == DirVerdict::kNoPeer)
        return out + "no peer access (not judged; such copies are refused)";
    if (!r.untested_reason.empty())
        return out + "untested (" + r.untested_reason + ")";
    if (r.ce.ran && !r.ce.corrupt && !r.ce.untested) {
        char ce[96];
        snprintf(ce, sizeof(ce),
                 "copy engines OK (%d x %" PRIu64 " KiB, %.0f MB/s)",
                 kSelfTestIterations, r.bytes >> 10,
                 r.ce_us ? double(r.ce_bytes) / double(r.ce_us) : 0.0);
        out += ce;
    } else {
        out += describePath("copy engines", r.ce);
    }
    out += ", ";
    if (r.kernel.ran || r.kernel.untested || r.kernel.corrupt)
        out += describePath(
            ("gather kernel on gpu " +
             std::to_string(r.kernel.ran
                                ? r.kernel_dev
                                : (g_cfg.gather_on_src ? r.src : r.dst)))
                .c_str(),
            r.kernel);
    else
        out += "gather kernel not used";
    return out + ": " + dirVerdictName(r.verdict());
}

// Host-path self-test of one GPU (what host-staged copies use): copy
// engines device -> pinned host -> device, and the gather kernel writing into
// and reading from pinned host memory, each read back and compared. Only a
// mismatch fails it.
DirVerdict hostSelfTest(int dev, std::string *detail) {
    const uint64_t n = 1ull << 20;
    char *dbuf = nullptr;
    void *h[3] = {nullptr, nullptr, nullptr};
    cudaError_t e = cudaSetDevice(dev);
    if (e == cudaSuccess) e = cudaMalloc(reinterpret_cast<void **>(&dbuf), n);
    for (int i = 0; i < 3 && e == cudaSuccess; ++i)
        e = cudaHostAlloc(&h[i], n,
                          cudaHostAllocPortable | cudaHostAllocMapped);
    DirVerdict v = DirVerdict::kPass;
    uint8_t *pat = static_cast<uint8_t *>(h[0]);  // source pattern
    uint8_t *out = static_cast<uint8_t *>(h[1]);  // read back
    uint8_t *aux = static_cast<uint8_t *>(h[2]);  // second pattern / poison
    std::vector<uint8_t> expect(n), poison(n);
    auto check = [&](const uint8_t *want, const char *what) {
        const Mismatch m = compareImages(want, out, n);
        if (!m.any()) return true;
        char why[192];
        snprintf(why, sizeof(why),
                 "%s CORRUPT (%" PRIu64 " bytes wrong, first at +%#" PRIx64 ")",
                 what, m.bytes, m.first);
        *detail = why;
        v = DirVerdict::kFail;
        return false;
    };
    DeviceResources *res = nullptr;
    if (e == cudaSuccess) e = t_res.get(dev, &res);
    const bool kernel = g_cfg.gather_threshold > 0;
    for (int iter = 0; iter < 2 && e == cudaSuccess && v == DirVerdict::kPass;
         ++iter) {
        const uint64_t seed = mixSeed(g_epoch ^ (uint64_t(dev) << 40), iter);
        // Copy engines: host -> device -> host.
        fillPattern(pat, n, seed);
        fillPattern(out, n, ~seed);
        e = cudaMemcpyAsync(dbuf, pat, n, cudaMemcpyHostToDevice, res->stream);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(out, dbuf, n, cudaMemcpyDeviceToHost,
                                res->stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(res->stream);
        if (e != cudaSuccess || !check(pat, "copy engines")) break;
        if (!kernel || g_kernel_unusable[dev].load()) continue;
        // Gather kernel writing into host memory (the sending side):
        // device (holds |pat|) -> |out| (poisoned).
        const std::vector<TestCopy> plan =
            selfTestPlan(n, uint32_t(iter), false, g_cfg.gather_threshold);
        fillPattern(poison.data(), n, seed * 3 + 1);
        memcpy(out, poison.data(), n);
        expectedImage(pat, poison.data(), n, plan, expect.data());
        std::vector<GatherEntry> table;
        for (const TestCopy &c : plan)
            table.push_back({reinterpret_cast<uint64_t>(out + c.dst_off),
                             reinterpret_cast<uint64_t>(dbuf + c.src_off),
                             c.len});
        e = t_res.reserve(dev, *res, table.size());
        if (e == cudaSuccess) {
            memcpy(res->host_table, table.data(),
                   table.size() * sizeof(GatherEntry));
            e = cudaMemcpyAsync(res->dev_table, res->host_table,
                                table.size() * sizeof(GatherEntry),
                                cudaMemcpyHostToDevice, res->stream);
        }
        if (e == cudaSuccess)
            e = launchGatherCopy(res->dev_table, uint32_t(table.size()),
                                 res->grid_blocks, res->stream);
        if (e == cudaErrorNoKernelImageForDevice ||
            e == cudaErrorInvalidDeviceFunction ||
            e == cudaErrorUnsupportedPtxVersion) {
            cudaGetLastError();  // production uses the copy engines there
            g_kernel_unusable[dev] = true;
            e = cudaStreamSynchronize(res->stream);
            continue;
        }
        if (e == cudaSuccess) e = cudaStreamSynchronize(res->stream);
        if (e != cudaSuccess || !check(expect.data(), "kernel to host")) break;
        // Gather kernel reading host memory (the receiving side): |aux| (a
        // second pattern) -> device (holds |pat|) -> read back.
        fillPattern(aux, n, seed * 5 + 2);
        expectedImage(aux, pat, n, plan, expect.data());
        table.clear();
        for (const TestCopy &c : plan)
            table.push_back({reinterpret_cast<uint64_t>(dbuf + c.dst_off),
                             reinterpret_cast<uint64_t>(aux + c.src_off),
                             c.len});
        memcpy(res->host_table, table.data(),
               table.size() * sizeof(GatherEntry));
        e = cudaMemcpyAsync(res->dev_table, res->host_table,
                            table.size() * sizeof(GatherEntry),
                            cudaMemcpyHostToDevice, res->stream);
        if (e == cudaSuccess)
            e = launchGatherCopy(res->dev_table, uint32_t(table.size()),
                                 res->grid_blocks, res->stream);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(out, dbuf, n, cudaMemcpyDeviceToHost,
                                res->stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(res->stream);
        if (e != cudaSuccess || !check(expect.data(), "kernel from host"))
            break;
    }
    if (e != cudaSuccess) {
        if (isStickyCudaError(e)) dieOnStickyError(e, "host-path self-test");
        cudaGetLastError();
        *detail = std::string("untested: ") + cudaGetErrorString(e);
        v = DirVerdict::kUntested;
    }
    cudaSetDevice(dev);
    for (void *p : h)
        if (p) cudaFreeHost(p);
    if (dbuf) cudaFree(dbuf);
    cudaGetLastError();
    return v;
}

// Staged-copy self-test of one direction: the real stagedCopy() from |s| to
// |d| (copy-engine entries in one half of the buffers, kernel-sized entries
// in the other, more than one staging piece), the destination read back and
// compared. Only a mismatch fails it.
DirVerdict stageSelfTestDirection(int s, int d, std::string *detail) {
    const uint64_t half = std::max<uint64_t>(
        2ull << 20, selfTestMinBufferBytes(g_cfg.gather_threshold));
    const uint64_t n = 2 * half;
    char *sb = nullptr, *db = nullptr;
    cudaError_t e = cudaSetDevice(s);
    if (e == cudaSuccess) e = cudaMalloc(reinterpret_cast<void **>(&sb), n);
    if (e == cudaSuccess) e = cudaSetDevice(d);
    if (e == cudaSuccess) e = cudaMalloc(reinterpret_cast<void **>(&db), n);
    DirVerdict v = DirVerdict::kPass;
    std::vector<uint8_t> src(n), poison(n), expect(n), got(n);
    for (int iter = 0; iter < 2 && e == cudaSuccess && v == DirVerdict::kPass;
         ++iter) {
        const uint64_t seed = mixSeed(
            g_epoch ^ (uint64_t(s) << 40) ^ (uint64_t(d) << 32), 77 + iter);
        fillPattern(src.data(), n, seed);
        fillPattern(poison.data(), n, ~seed);
        e = uploadSync(s, sb, src.data(), n);
        if (e == cudaSuccess) e = uploadSync(d, db, poison.data(), n);
        if (e != cudaSuccess) break;
        std::vector<TestCopy> plan =
            selfTestPlan(half, uint32_t(iter), true, g_cfg.gather_threshold);
        for (TestCopy c : selfTestPlan(half, uint32_t(iter), false,
                                       g_cfg.gather_threshold)) {
            c.src_off += half;
            c.dst_off += half;
            plan.push_back(c);
        }
        std::vector<StagedItem> items;
        for (const TestCopy &c : plan)
            items.push_back({sb + c.src_off, db + c.dst_off, c.len});
        std::string err;
        const uint64_t now = nowUs();
        if (!stagedCopy(s, d, items, now + 5000000, now + 20000000, &err)) {
            *detail = "untested: " + err;
            v = DirVerdict::kUntested;
            break;
        }
        e = cudaSetDevice(d);
        if (e == cudaSuccess)
            e = cudaMemcpy(got.data(), db, n, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) break;
        expectedImage(src.data(), poison.data(), n, plan, expect.data());
        const Mismatch m = compareImages(expect.data(), got.data(), n);
        if (m.any()) {
            char why[160];
            snprintf(why, sizeof(why),
                     "CORRUPT (%" PRIu64 " bytes wrong, first at +%#" PRIx64
                     ")",
                     m.bytes, m.first);
            *detail = why;
            v = DirVerdict::kFail;
        }
    }
    if (e != cudaSuccess) {
        if (isStickyCudaError(e)) dieOnStickyError(e, "staged-copy self-test");
        cudaGetLastError();
        *detail = std::string("untested: ") + cudaGetErrorString(e);
        v = DirVerdict::kUntested;
    }
    if (sb) {
        cudaSetDevice(s);
        cudaFree(sb);
    }
    if (db) {
        cudaSetDevice(d);
        cudaFree(db);
    }
    cudaGetLastError();
    return v;
}

// --p2p-selftest-child: tests |pairs_arg| ("<bus>+<bus>,...": a GPU's host
// path when both are the same GPU, else the pair's P2P paths and -- with
// --p2p-fallback=host -- its host-staged copies) and |stage_arg| (pairs whose
// host-staged copies only are tested), and reports on stdout (see
// p2p_policy.h); logs go to stderr like the daemon's.
int runSelfTestChild(const std::string &pairs_arg,
                     const std::string &stage_arg) {
    std::vector<PciPair> pairs, stage_pairs;
    if ((!pairs_arg.empty() && !parsePciPairs(pairs_arg, pairs)) ||
        (!stage_arg.empty() && !parsePciPairs(stage_arg, stage_pairs))) {
        LOGE("self-test child: bad pair list");
        return 2;
    }
    if (!initDevices(/*quiet=*/true)) return 3;
    {
        std::random_device rd;
        g_epoch = (uint64_t(rd()) << 32) ^ rd() ^ uint64_t(getpid());
    }
    // A small pool of its own for the staged-copy test (1 MiB slots).
    g_host_staging_bytes = 4ull << 20;
    auto find = [](const PciAddr &a) {
        for (const Device &d : g_devices)
            if (d.addr == a) return d.ordinal;
        return -1;
    };
    std::map<int, bool> ctx;
    auto context = [&](int d) {
        auto it = ctx.find(d);
        if (it != ctx.end()) return it->second;
        cudaError_t e = cudaSetDevice(d);
        if (e == cudaSuccess) e = cudaFree(nullptr);
        if (e != cudaSuccess) {
            if (isStickyCudaError(e)) dieOnStickyError(e, "P2P self-test");
            cudaGetLastError();
        }
        return ctx[d] = e == cudaSuccess;
    };
    std::set<int> kernel_reported;
    auto reportKernel = [&]() {
        for (const Device &dv : g_devices) {
            if (!g_kernel_unusable[dv.ordinal].load() ||
                !kernel_reported.insert(dv.ordinal).second)
                continue;
            printf("NOKERNEL %s %s\n", formatPciAddr(dv.addr).c_str(),
                   formatPciAddr(dv.addr).c_str());
        }
    };
    auto stageBoth = [&](int a, int b, const std::string &sa,
                         const std::string &sb) {
        DirVerdict v[2];
        for (int k = 0; k < 2; ++k) {
            const int s = k ? b : a, d = k ? a : b;
            std::string detail;
            v[k] = stageSelfTestDirection(s, d, &detail);
            if (v[k] == DirVerdict::kPass)
                LOGI(
                    "P2P self-test gpu %d (%s) -> gpu %d (%s): host-staged "
                    "copy OK: PASS",
                    s, g_devices[s].pci.c_str(), d, g_devices[d].pci.c_str());
            else
                LOGW(
                    "P2P self-test gpu %d (%s) -> gpu %d (%s): host-staged "
                    "copy %s: %s",
                    s, g_devices[s].pci.c_str(), d, g_devices[d].pci.c_str(),
                    detail.c_str(), dirVerdictName(v[k]));
        }
        printf("STAGE %s %s %s\nSTAGE %s %s %s\n", sa.c_str(), sb.c_str(),
               dirVerdictName(v[0]), sb.c_str(), sa.c_str(),
               dirVerdictName(v[1]));
    };
    for (const PciPair &p : pairs) {
        const std::string sa = formatPciAddr(p.first),
                          sb = formatPciAddr(p.second);
        printf("BEGIN %s %s\n", sa.c_str(), sb.c_str());
        fflush(stdout);
        const int a = find(p.first), b = find(p.second);
        DirVerdict v[2] = {DirVerdict::kUntested, DirVerdict::kUntested};
        if (a >= 0 && a == b) {  // host-path check of one GPU
            std::string detail;
            const DirVerdict hv =
                context(a) ? hostSelfTest(a, &detail) : DirVerdict::kUntested;
            if (hv == DirVerdict::kPass)
                LOGI(
                    "P2P self-test gpu %d (%s) host path: copy engines and "
                    "gather kernel OK: PASS",
                    a, sa.c_str());
            else
                LOGW("P2P self-test gpu %d (%s) host path: %s: %s", a,
                     sa.c_str(),
                     detail.empty() ? "no CUDA context" : detail.c_str(),
                     dirVerdictName(hv));
            reportKernel();
            printf("DIR %s %s %s\nEND %s %s\n", sa.c_str(), sa.c_str(),
                   dirVerdictName(hv), sa.c_str(), sa.c_str());
            fflush(stdout);
            continue;
        }
        const bool usable = a >= 0 && b >= 0 && a != b;
        if (!usable) {
            LOGW("P2P self-test %s <-> %s: not two GPUs visible to the test",
                 sa.c_str(), sb.c_str());
        } else if (!context(a) || !context(b)) {
            LOGW(
                "P2P self-test gpu %d (%s) <-> gpu %d (%s): untested (no "
                "CUDA context)",
                a, sa.c_str(), b, sb.c_str());
        } else {
            enablePeer(a, b);
            enablePeer(b, a);
            const std::vector<DirectionResult> r =
                runSelfTests({{a, b}, {b, a}});
            for (int k = 0; k < 2; ++k) {
                v[k] = r[k].verdict();
                const std::string line = describeDirection(r[k]);
                if (v[k] == DirVerdict::kPass || v[k] == DirVerdict::kNoPeer)
                    LOGI("P2P self-test %s", line.c_str());
                else
                    LOGW("P2P self-test %s", line.c_str());
            }
        }
        printf("DIR %s %s %s\nDIR %s %s %s\n", sa.c_str(), sb.c_str(),
               dirVerdictName(v[0]), sb.c_str(), sa.c_str(),
               dirVerdictName(v[1]));
        if (g_fallback_host && usable && ctx[a] && ctx[b])
            stageBoth(a, b, sa, sb);
        reportKernel();
        printf("END %s %s\n", sa.c_str(), sb.c_str());
        fflush(stdout);
    }
    for (const PciPair &p : stage_pairs) {
        const std::string sa = formatPciAddr(p.first),
                          sb = formatPciAddr(p.second);
        printf("BEGIN %s %s\n", sa.c_str(), sb.c_str());
        fflush(stdout);
        const int a = find(p.first), b = find(p.second);
        if (a >= 0 && b >= 0 && a != b && context(a) && context(b))
            stageBoth(a, b, sa, sb);
        reportKernel();
        printf("END %s %s\n", sa.c_str(), sb.c_str());
        fflush(stdout);
    }
    return 0;
}

bool enablePeer(int dst, int src) {
    int can = 0;
    cudaDeviceCanAccessPeer(&can, dst, src);
    if (!can) {
        LOGW(
            "gpu %d cannot access gpu %d as a peer; copies between them are "
            "refused (the engines fall back to their base transport)",
            dst, src);
        return false;
    }
    cudaError_t e = cudaSetDevice(dst);
    if (e == cudaSuccess) e = cudaDeviceEnablePeerAccess(src, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        e = cudaSuccess;
    }
    if (e != cudaSuccess) {
        LOGW("enable peer access gpu %d -> gpu %d: %s", dst, src,
             cudaGetErrorString(e));
        cudaGetLastError();
        return false;
    }
    g_peer[dst][src] = true;
    return true;
}

// Create |dev|'s context and enable peer access between it and every GPU
// initialized before, in both directions. This happens before any IPC
// handle that involves |dev| is opened: a handle opened in a context without
// peer access to the owning GPU is staged through host memory.
bool ensureDevice(int dev) {
    if (g_dev_ready[dev].load(std::memory_order_acquire)) return true;
    std::lock_guard<std::mutex> lock(g_dev_mu);
    if (g_dev_ready[dev].load()) return true;
    cudaError_t e = cudaSetDevice(dev);
    if (e == cudaSuccess) e = cudaFree(nullptr);  // create the context
    if (e != cudaSuccess) {
        LOGE("initializing gpu %d: %s%s", dev, cudaGetErrorString(e),
             e == cudaErrorDevicesUnavailable
                 ? " (exclusive compute mode and the GPU is in use; the "
                   "proxy needs compute mode DEFAULT)"
                 : "");
        cudaGetLastError();
        return false;
    }
    int peers = 0;
    std::string list;
    for (size_t o = 0; o < g_devices.size(); ++o) {
        int other = static_cast<int>(o);
        if (other == dev || !g_dev_ready[other].load()) continue;
        bool a = enablePeer(dev, other);
        bool b = enablePeer(other, dev);
        if (a && b) {
            ++peers;
            list += " " + std::to_string(other);
        }
    }
    g_dev_ready[dev].store(true, std::memory_order_release);
    LOGI("gpu %d initialized; peer access with %d gpu(s):%s", dev, peers,
         list.empty() ? " none yet" : list.c_str());
    return true;
}

// -------------------------------------------- P2P self-test (daemon side)
// The daemon never runs a test itself: it re-executes its own binary with
// --p2p-selftest-child for the items to test (posix_spawn, so nothing CUDA is
// forked) and reads the verdicts from a pipe, killing the child when it makes
// no progress for --p2p-selftest-timeout seconds. An item the child was
// testing when it died (an unrecoverable CUDA error, a crash) or hung is
// failed. In enforce mode a pair is refused for P2P from the moment it is
// requested until a test passes; a test that could not run is repeated with
// a backoff and at the next registration on either GPU.
//
// Tester items: (a, a) the host path of GPU a; (a, b) with a < b the pair's
// P2P paths (and, with --p2p-fallback=host, its host-staged copies); (b, a)
// with b > a the pair's host-staged copies only (a pair whose P2P verdict is
// already final or that is denied on the command line).
std::atomic<pid_t> g_selftest_child{0};

enum class ItemKind { kHost, kPair, kStage };
ItemKind itemKind(const std::pair<int, int> &p) {
    return p.first == p.second  ? ItemKind::kHost
           : p.first < p.second ? ItemKind::kPair
                                : ItemKind::kStage;
}

// What the child reported for one item.
struct ItemReport {
    std::pair<int, int> item{-1, -1};
    DirVerdict ab = DirVerdict::kUntested, ba = DirVerdict::kUntested;
    bool staged_seen = false;
    DirVerdict sab = DirVerdict::kUntested, sba = DirVerdict::kUntested;
};
using ItemReportFn = std::function<void(const ItemReport &)>;

struct ChildOutcome {
    bool began = false;                       // reported at least one BEGIN
    std::pair<int, int> in_progress{-1, -1};  // item without END
    std::string death;                        // empty: exited 0 when done
};

int ordinalOf(const PciAddr &a) {
    for (const Device &d : g_devices)
        if (d.addr == a) return d.ordinal;
    return -1;
}

void reapChild(pid_t pid, std::string *death) {
    int status = 0;
    const uint64_t until = nowUs() + 5000000;
    pid_t got = 0;
    while ((got = waitpid(pid, &status, WNOHANG)) == 0 && nowUs() < until)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (got == 0) {
        // Stuck in the driver: reap it whenever it goes.
        try {
            std::thread([pid]() { waitpid(pid, nullptr, 0); }).detach();
        } catch (const std::system_error &) {
        }
        if (death->empty()) *death = "did not exit";
        return;
    }
    if (!death->empty() || got != pid) return;
    if (WIFSIGNALED(status)) {
        *death = "killed by signal " + std::to_string(WTERMSIG(status));
    } else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        *death =
            "exit status " + std::to_string(WEXITSTATUS(status)) +
            (WEXITSTATUS(status) == 70 ? " (unrecoverable CUDA error)" : "");
    }
}

// Runs one child over |items| (host and pair items first, then stage items:
// the order the child tests them in); |report| gets each item as soon as the
// child finished it.
ChildOutcome runSelfTestChildProcess(
    const std::vector<std::pair<int, int>> &items, const ItemReportFn &report) {
    ChildOutcome out;
    std::vector<PciPair> pci, stage_pci;
    std::set<std::pair<int, int>> full;  // (min, max) of host and pair items
    for (const auto &p : items) {
        const int lo = std::min(p.first, p.second),
                  hi = std::max(p.first, p.second);
        if (itemKind(p) == ItemKind::kStage) {
            stage_pci.push_back({g_devices[lo].addr, g_devices[hi].addr});
        } else {
            pci.push_back({g_devices[lo].addr, g_devices[hi].addr});
            full.insert({lo, hi});
        }
    }
    std::vector<std::string> args = {
        g_self_exe,           "--p2p-selftest-child",
        "--gather-threshold", std::to_string(g_cfg.gather_threshold),
        "--gather-on",        g_cfg.gather_on_src ? "src" : "dst",
        "--p2p-fallback",     g_fallback_host ? "host" : "tcp"};
    if (!pci.empty()) {
        args.push_back("--p2p-selftest-pairs");
        args.push_back(encodePciPairs(pci));
    }
    if (!stage_pci.empty()) {
        args.push_back("--p2p-selftest-stage-pairs");
        args.push_back(encodePciPairs(stage_pci));
    }
    std::vector<char *> argv;
    for (auto &a : args) argv.push_back(&a[0]);
    argv.push_back(nullptr);
    int fds[2];
    if (pipe2(fds, O_CLOEXEC) != 0) {
        out.death = std::string("pipe: ") + strerror(errno);
        return out;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], STDOUT_FILENO);
    pid_t pid = 0;
    const int rc = posix_spawn(&pid, g_self_exe.c_str(), &fa, nullptr,
                               argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    close(fds[1]);
    if (rc != 0) {
        close(fds[0]);
        out.death =
            std::string("cannot start ") + g_self_exe + ": " + strerror(rc);
        return out;
    }
    g_selftest_child = pid;
    ItemReport cur;
    bool full_phase = !pci.empty();
    std::string buf;
    uint64_t deadline = nowUs() + uint64_t(g_selftest_timeout_s) * 1000000;
    while (true) {
        const uint64_t now = nowUs();
        if (now >= deadline) {
            kill(pid, SIGKILL);
            out.death = "no progress for " +
                        std::to_string(g_selftest_timeout_s) + " s, killed";
            break;
        }
        pollfd p{fds[0], POLLIN, 0};
        const int prc = poll(
            &p, 1, int(std::min<uint64_t>((deadline - now) / 1000 + 1, 1000)));
        if (prc < 0 && errno == EINTR) continue;
        if (prc <= 0) continue;
        char chunk[4096];
        const ssize_t n = read(fds[0], chunk, sizeof(chunk));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;  // EOF: the child is gone
        buf.append(chunk, size_t(n));
        size_t nl;
        while ((nl = buf.find('\n')) != std::string::npos) {
            const ChildLine line = parseChildLine(buf.substr(0, nl));
            buf.erase(0, nl + 1);
            const int a = ordinalOf(line.a), b = ordinalOf(line.b);
            if (line.kind == ChildLine::kInvalid || a < 0 || b < 0) continue;
            deadline = nowUs() + uint64_t(g_selftest_timeout_s) * 1000000;
            switch (line.kind) {
                case ChildLine::kBegin: {
                    const auto lohi =
                        std::make_pair(std::min(a, b), std::max(a, b));
                    // The child tests the pair list before the stage list.
                    if (full_phase && !full.count(lohi)) full_phase = false;
                    cur = ItemReport();
                    cur.item = full_phase || a == b
                                   ? lohi
                                   : std::make_pair(lohi.second, lohi.first);
                    out.began = true;
                    out.in_progress = cur.item;
                    break;
                }
                case ChildLine::kDir:
                    (a <= b ? cur.ab : cur.ba) = line.verdict;
                    break;
                case ChildLine::kStage:
                    cur.staged_seen = true;
                    (a < b ? cur.sab : cur.sba) = line.verdict;
                    break;
                case ChildLine::kNoKernel:
                    if (!g_kernel_unusable[a].exchange(true))
                        LOGW(
                            "gather kernel unavailable on gpu %d (%s); small "
                            "copies use the copy engines",
                            a, g_devices[a].pci.c_str());
                    break;
                case ChildLine::kEnd:
                    out.in_progress = {-1, -1};
                    report(cur);
                    break;
                default:
                    break;
            }
        }
    }
    close(fds[0]);
    reapChild(pid, &out.death);
    g_selftest_child = 0;
    return out;
}

class SelfTester {
   public:
    void start() {
        try {
            std::thread(&SelfTester::loop, this).detach();
        } catch (const std::system_error &e) {
            LOGE(
                "cannot start the P2P self-test thread (%s): pairs stay "
                "untested",
                e.what());
        }
    }

    // Items to test unless already decided, queued, or not needed. Pairs
    // come as (x, y) in any order: the P2P test while the pair's P2P verdict
    // is open, else (and for a pair denied on the command line) the
    // host-staged copy test while that verdict is open. In enforce mode a
    // pair under P2P test is refused for P2P until it passes.
    void request(const std::vector<std::pair<int, int>> &asked) {
        if (g_selftest == SelfTestMode::kOff) return;
        std::lock_guard<std::mutex> lock(mu_);
        bool added = false;
        auto queue = [&](std::pair<int, int> item) {
            if (pending_.count(item)) return;
            queue_.push_back(item);
            pending_.insert(item);
            added = true;
        };
        for (const auto &p : asked) {
            const int a = std::min(p.first, p.second),
                      b = std::max(p.first, p.second);
            if (a == b) {  // host path of one GPU
                if (g_fallback_host && g_host_state[a] == kNotTested)
                    queue({a, a});
                continue;
            }
            const bool stat = g_policy.reason(a, b) == PeerPolicy::kStatic &&
                              g_policy.reason(b, a) == PeerPolicy::kStatic;
            if (!stat && g_test_state[a][b] == kNotTested) {
                if (pending_.count({b, a})) continue;  // stage test queued
                if (g_selftest == SelfTestMode::kEnforce)
                    g_policy.setPair(a, b, PeerPolicy::kUntested);
                queue({a, b});
            } else if (g_fallback_host && g_stage_state[a][b] == kNotTested &&
                       !pending_.count({a, b})) {
                queue({b, a});
            }
        }
        if (added) cv_.notify_all();
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        const pid_t pid = g_selftest_child.load();
        if (pid > 0) kill(pid, SIGKILL);
    }

   private:
    // An item whose test could not run (typically no GPU memory for the
    // child next to the engines) is retried here, besides at the next
    // registration on either GPU: a running engine never registers again.
    struct Retry {
        int attempts = 0;
        uint64_t next_us = 0;
    };

    static uint64_t retryDelayS(int attempts) {
        return attempts <= 1   ? 10
               : attempts == 2 ? 30
               : attempts == 3 ? 60
                               : 300;
    }

    static uint8_t itemState(const std::pair<int, int> &p) {
        const int lo = std::min(p.first, p.second),
                  hi = std::max(p.first, p.second);
        switch (itemKind(p)) {
            case ItemKind::kHost:
                return g_host_state[lo].load();
            case ItemKind::kPair:
                return g_test_state[lo][hi].load();
            case ItemKind::kStage:
                return g_stage_state[lo][hi].load();
        }
        return kNotTested;
    }

    // Under mu_.
    void countUntested() {
        size_t hosts = 0, pairs = 0, stages = 0;
        for (const auto &kv : retry_) {
            switch (itemKind(kv.first)) {
                case ItemKind::kHost:
                    ++hosts;
                    break;
                case ItemKind::kPair:
                    ++pairs;
                    break;
                case ItemKind::kStage:
                    ++stages;
                    break;
            }
        }
        g_untested_host_paths = hosts;
        g_untested_pairs = pairs;
        g_untested_staged_pairs = stages;
    }

    // Under mu_: queues the retries that are due; returns the time of the
    // next one (0: none).
    uint64_t queueDueRetries() {
        const uint64_t now = nowUs();
        uint64_t next = 0;
        for (auto it = retry_.begin(); it != retry_.end();) {
            const auto p = it->first;
            if (itemState(p) != kNotTested) {
                it = retry_.erase(it);  // decided meanwhile
                continue;
            }
            if (!pending_.count(p)) {
                if (it->second.next_us <= now) {
                    queue_.push_back(p);
                    pending_.insert(p);
                } else if (!next || it->second.next_us < next) {
                    next = it->second.next_us;
                }
            }
            ++it;
        }
        countUntested();
        return next;
    }

    void loop() {
        while (true) {
            std::vector<std::pair<int, int>> batch;
            {
                std::unique_lock<std::mutex> lock(mu_);
                while (!stop_ && queue_.empty()) {
                    const uint64_t next = queueDueRetries();
                    if (!queue_.empty()) break;
                    const uint64_t now = nowUs();
                    if (next)
                        cv_.wait_for(lock, std::chrono::microseconds(
                                               next > now ? next - now : 0));
                    else
                        cv_.wait(lock);
                }
                if (stop_) return;
                batch.swap(queue_);
            }
            runBatch(batch);
            std::lock_guard<std::mutex> lock(mu_);
            for (const auto &p : batch) pending_.erase(p);
        }
    }

    // Records the outcome of |item| for the retry schedule; returns the
    // delay until its next retry when it stays untested (0 otherwise).
    uint64_t noteOutcome(const std::pair<int, int> &item, bool untested,
                         int *attempt) {
        std::lock_guard<std::mutex> lock(mu_);
        uint64_t delay = 0;
        if (untested) {
            Retry &r = retry_[item];
            ++r.attempts;
            delay = retryDelayS(r.attempts);
            r.next_us = nowUs() + delay * 1000000;
            *attempt = r.attempts;
        } else {
            retry_.erase(item);
        }
        countUntested();
        return delay;
    }

    static uint8_t stateOf(PairVerdict v) {
        return v == PairVerdict::kPassed   ? kTestPassed
               : v == PairVerdict::kFailed ? kTestFailed
                                           : kNotTested;
    }

    // Where the copies of a pair refused for P2P go.
    static std::string route(int a, int b) {
        if (!g_fallback_host) return "TCP (--p2p-fallback=tcp)";
        for (int x : {a, b}) {
            if (g_host_state[x] != kTestPassed)
                return "TCP (host path of gpu " + std::to_string(x) +
                       (g_host_state[x] == kTestFailed ? " failed)"
                                                       : " not verified yet)");
        }
        const uint8_t st = g_stage_state[std::min(a, b)][std::max(a, b)];
        if (st != kTestPassed)
            return st == kTestFailed
                       ? "TCP (host-staged copy failed its test)"
                       : "TCP (host-staged copy not verified yet)";
        return "host-staged";
    }

    void applyHost(int a, PairVerdict v, const std::string &why) {
        char head[128];
        snprintf(head, sizeof(head), "P2P self-test gpu %d (%s) host path", a,
                 g_devices[a].pci.c_str());
        const std::string reason = why.empty() ? "" : " (" + why + ")";
        g_host_state[a] = stateOf(v);
        int attempt = 0;
        const uint64_t delay =
            noteOutcome({a, a}, v == PairVerdict::kUntested, &attempt);
        switch (v) {
            case PairVerdict::kPassed:
                LOGI("%s: passed", head);
                break;
            case PairVerdict::kFailed:
                LOGW("%s: FAILED%s -> pairs with it are never host-staged",
                     head, reason.c_str());
                break;
            case PairVerdict::kUntested:
                LOGW("%s: untested%s; attempt %d, tested again in %" PRIu64
                     " s",
                     head, reason.c_str(), attempt, delay);
                break;
        }
    }

    void applyStage(int a, int b, PairVerdict v, const std::string &why) {
        if (!g_fallback_host) return;
        char head[160];
        snprintf(head, sizeof(head),
                 "P2P self-test gpu %d (%s) <-> gpu %d (%s) host-staged copies",
                 a, g_devices[a].pci.c_str(), b, g_devices[b].pci.c_str());
        const std::string reason = why.empty() ? "" : " (" + why + ")";
        g_stage_state[a][b] = stateOf(v);
        int attempt = 0;
        const uint64_t delay =
            noteOutcome({b, a}, v == PairVerdict::kUntested, &attempt);
        switch (v) {
            case PairVerdict::kPassed:
                LOGI("%s: passed", head);
                break;
            case PairVerdict::kFailed:
                LOGW(
                    "%s: FAILED%s -> what the pair cannot copy over P2P uses "
                    "TCP",
                    head, reason.c_str());
                break;
            case PairVerdict::kUntested:
                LOGW("%s: untested%s; attempt %d, tested again in %" PRIu64
                     " s",
                     head, reason.c_str(), attempt, delay);
                break;
        }
    }

    void apply(int a, int b, PairVerdict v, const std::string &why) {
        const bool enforce = g_selftest == SelfTestMode::kEnforce;
        char head[160];
        snprintf(head, sizeof(head),
                 "P2P self-test gpu %d (%s) <-> gpu %d (%s)", a,
                 g_devices[a].pci.c_str(), b, g_devices[b].pci.c_str());
        const std::string reason = why.empty() ? "" : " (" + why + ")";
        g_test_state[a][b] = g_test_state[b][a] = stateOf(v);
        int attempt = 0;
        const uint64_t delay =
            noteOutcome({a, b}, v == PairVerdict::kUntested, &attempt);
        switch (v) {
            case PairVerdict::kPassed:
                g_policy.setPair(a, b, PeerPolicy::kAllowed);
                LOGI("%s: passed -> allowed", head);
                break;
            case PairVerdict::kFailed:
                if (enforce)
                    g_policy.setPair(a, b, PeerPolicy::kSelfTestFailed);
                LOGW("%s: FAILED%s -> %s", head, reason.c_str(),
                     enforce ? ("P2P denied -> " + route(a, b)).c_str()
                             : "would be denied (warn mode: still used)");
                break;
            case PairVerdict::kUntested:
                if (enforce) g_policy.setPair(a, b, PeerPolicy::kUntested);
                LOGW(
                    "%s: untested%s -> %s; attempt %d, tested again in "
                    "%" PRIu64 " s or at the next registration on either GPU",
                    head, reason.c_str(),
                    enforce
                        ? ("P2P denied until a test passes -> " + route(a, b))
                              .c_str()
                        : "still used (warn mode)",
                    attempt, delay);
                break;
        }
    }

    void runBatch(std::vector<std::pair<int, int>> remaining) {
        const uint64_t t0 = nowUs();
        // A pair's P2P test also tests its host-staged copies: drop a
        // staged-copy-only item of the same pair (e.g. two retries due at
        // once), so that the child reports each pair once.
        {
            std::set<std::pair<int, int>> full;
            for (const auto &p : remaining)
                if (itemKind(p) == ItemKind::kPair) full.insert(p);
            remaining.erase(
                std::remove_if(remaining.begin(), remaining.end(),
                               [&](const std::pair<int, int> &p) {
                                   return itemKind(p) == ItemKind::kStage &&
                                          full.count({p.second, p.first});
                               }),
                remaining.end());
        }
        // Host paths first (the pair verdicts then say where refused pairs'
        // copies go), then pairs, then staged-copy-only items: the order the
        // child tests them in.
        std::stable_sort(
            remaining.begin(), remaining.end(),
            [](const std::pair<int, int> &x, const std::pair<int, int> &y) {
                return int(itemKind(x)) < int(itemKind(y));
            });
        size_t hosts = 0, pairs = 0, stages = 0;
        for (const auto &p : remaining) {
            const ItemKind k = itemKind(p);
            (k == ItemKind::kHost   ? hosts
             : k == ItemKind::kPair ? pairs
                                    : stages)++;
        }
        size_t passed = 0, failed = 0, untested = 0;
        auto record = [&](const ItemReport &r, bool died,
                          const std::string &why) {
            const int lo = std::min(r.item.first, r.item.second),
                      hi = std::max(r.item.first, r.item.second);
            const PairVerdict staged = died ? PairVerdict::kFailed
                                       : r.staged_seen
                                           ? combineVerdicts(r.sab, r.sba)
                                           : PairVerdict::kUntested;
            switch (itemKind(r.item)) {
                case ItemKind::kHost:
                    applyHost(lo,
                              died ? PairVerdict::kFailed
                                   : combineVerdicts(r.ab, r.ab),
                              why);
                    break;
                case ItemKind::kPair: {
                    // The staged-copy verdict first: the pair's line says
                    // where its copies go.
                    if (g_fallback_host) applyStage(lo, hi, staged, why);
                    const PairVerdict v = died ? PairVerdict::kFailed
                                               : combineVerdicts(r.ab, r.ba);
                    apply(lo, hi, v, why);
                    (v == PairVerdict::kPassed   ? passed
                     : v == PairVerdict::kFailed ? failed
                                                 : untested)++;
                    break;
                }
                case ItemKind::kStage:
                    applyStage(lo, hi, staged, why);
                    break;
            }
            remaining.erase(
                std::remove(remaining.begin(), remaining.end(), r.item),
                remaining.end());
        };
        while (!remaining.empty()) {
            const std::vector<std::pair<int, int>> asked = remaining;
            const ChildOutcome out = runSelfTestChildProcess(
                asked, [&](const ItemReport &r) { record(r, false, ""); });
            if (out.in_progress.first >= 0) {
                ItemReport r;
                r.item = out.in_progress;
                record(r, true, "self-test child " + out.death);
            }
            if (out.death.empty() || !out.began) {
                // Exited without reporting them, or failed before testing
                // anything: untested.
                if (!out.death.empty())
                    LOGE(
                        "P2P self-test child failed before testing anything: "
                        "%s",
                        out.death.c_str());
                const std::string why =
                    out.death.empty() ? "not reported by the self-test child"
                                      : "self-test child: " + out.death;
                while (!remaining.empty()) {
                    ItemReport r;
                    r.item = remaining.front();
                    record(r, false, why);
                }
            }
        }
        LOGI(
            "P2P self-test of %zu GPU pair(s), %zu host path(s) and %zu "
            "host-staged copy check(s) took %.1f s; pairs: %zu passed, %zu "
            "failed, %zu untested",
            pairs, hosts, stages, (nowUs() - t0) / 1e6, passed, failed,
            untested);
        // Node verdict over every pair tested so far.
        const int n = static_cast<int>(g_devices.size());
        size_t node_passed = 0, node_failed = 0;
        for (int x = 0; x < n; ++x) {
            for (int y = x + 1; y < n; ++y) {
                node_passed += g_test_state[x][y] == kTestPassed;
                node_failed += g_test_state[x][y] == kTestFailed;
            }
        }
        if (failed && node_failed && !node_passed) {
            LOGW(
                "P2P unusable on this node: %zu/%zu GPU pairs tested so far "
                "failed the self-test; %s",
                node_failed, node_failed,
                g_selftest != SelfTestMode::kEnforce
                    ? "warn mode: P2P is still used"
                : g_fallback_host
                    ? "their copies are host-staged where verified, else "
                      "they use the engines' base transport"
                    : "their copies use the engines' base transport");
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::pair<int, int>> queue_;
    std::set<std::pair<int, int>> pending_;
    std::map<std::pair<int, int>, Retry> retry_;
    bool stop_ = false;
};
SelfTester g_tester;

// An engine's first block on |dev|: its host path, and its pairs with the
// other GPUs that engines use.
void requestSelfTestForDevice(int dev) {
    std::vector<std::pair<int, int>> items = {{dev, dev}};
    for (int other = 0; other < static_cast<int>(g_devices.size()); ++other)
        if (other != dev && g_dev_ready[other].load())
            items.push_back({other, dev});
    g_tester.request(items);
}

std::string g_socket_path;

void onSignal(int) { g_stop = true; }

int connectTo(const std::string &path) {
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        close(fd);
        return -1;
    }
    memcpy(addr.sun_path, path.c_str(), path.size() + 1);
    if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    struct timeval tv{5, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    return fd;
}

// --stats / --ping: query a running daemon.
int queryDaemon(const std::string &path, MsgType type) {
    int fd = connectTo(path);
    if (fd < 0) {
        fprintf(stderr, "cannot connect to %s: %s\n", path.c_str(),
                strerror(errno));
        return 1;
    }
    if (!sendMsg(fd, type, 1, ProxyStatus::kOk, nullptr, 0)) {
        fprintf(stderr, "send failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }
    MsgHeader h;
    std::string body;
    if (!readAll(fd, &h, sizeof(h)) || h.magic != kMagic ||
        h.payload_len > kMaxPayload) {
        fprintf(stderr, "no valid reply from %s\n", path.c_str());
        close(fd);
        return 1;
    }
    body.resize(h.payload_len);
    if (h.payload_len && !readAll(fd, &body[0], h.payload_len)) {
        close(fd);
        return 1;
    }
    close(fd);
    if (type == MsgType::kPing && body.size() == sizeof(PingResp)) {
        PingResp p;
        memcpy(&p, body.data(), sizeof(p));
        printf("ok epoch=%016" PRIx64 " uptime_ms=%" PRIu64 "\n",
               p.daemon_epoch, p.uptime_ms);
    } else {
        printf("%s\n", body.c_str());
    }
    return h.status == 0 ? 0 : 1;
}

void usage(const char *argv0) {
    printf(
        "Usage: %s --socket <path> [options]\n"
        "\n"
        "Node-local GPU copy daemon for the Mooncake nvlink_proxy transport.\n"
        "Lets engines in containers that each see only their own GPU copy\n"
        "GPU to GPU over NVLink / PCIe P2P. Run it in a container that sees\n"
        "every GPU of the node (no GPU request, NVIDIA_VISIBLE_DEVICES=all,\n"
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility) and share the socket\n"
        "directory with the engine containers (e.g. a hostPath volume).\n"
        "Engines enable the transport with MC_NVLINK_PROXY_SOCKET=<path>.\n"
        "\n"
        "Options:\n"
        "  --socket <path>         unix socket to listen on (required; env\n"
        "                          MC_NVLINK_PROXY_SOCKET is used if unset)\n"
        "  --socket-mode <octal>   permissions of the socket file (default "
        "0666)\n"
        "  --stats-interval <sec>  log a counters line every <sec> seconds\n"
        "                          when something changed (default 60, 0 = "
        "off)\n"
        "  --stats                 print the counters of the daemon listening\n"
        "                          on --socket and exit\n"
        "  --ping                  exit 0 if the daemon on --socket answers\n"
        "                          (usable as a liveness probe)\n"
        "  --gather-threshold <B>  copies shorter than B bytes (after "
        "merging)\n"
        "                          run in one gather/scatter kernel per batch\n"
        "                          instead of the copy engines (default "
        "131072,\n"
        "                          0 = always use the copy engines)\n"
        "  --gather-on dst|src     GPU that runs the kernel: the receiving "
        "one\n"
        "                          (reads through the peer mapping, default) "
        "or\n"
        "                          the sending one (writes through it)\n"
        "  --no-coalesce           do not merge consecutive copies that are\n"
        "                          contiguous in source and destination\n"
        "  --drain-timeout-ms <ms> on SIGTERM/SIGINT, wait up to this long "
        "for\n"
        "                          running copies to finish (default 10000)\n"
        "  --p2p-selftest <mode>   before serving copies between two GPUs, a\n"
        "                          child process copies known patterns "
        "between\n"
        "                          them in both directions (copy engines and\n"
        "                          gather kernel) and verifies every byte.\n"
        "                          enforce (default): a pair is refused until\n"
        "                          its test passes, for good if the data came\n"
        "                          back wrong (or the child died or hung on\n"
        "                          it); a test that could not run (e.g. no "
        "GPU\n"
        "                          memory for it) is repeated after 10, 30,\n"
        "                          60 s, then every 5 min, and at the next\n"
        "                          registration on either GPU.\n"
        "                          warn: log only; off: no test\n"
        "  --no-startup-selftest   test a pair only when engines first\n"
        "                          register on both of its GPUs (recommended\n"
        "                          in production: GPUs without engines are\n"
        "                          never touched); by default every pair is\n"
        "                          also tested at start (a node audit)\n"
        "  --p2p-selftest-timeout <s>  kill a self-test child that makes no\n"
        "                          progress for this long (default 30)\n"
        "  --p2p-fallback host|tcp copies that may not use P2P (refused or\n"
        "                          no peer access): host (default) copies\n"
        "                          them through pinned host memory here, for\n"
        "                          pairs whose host paths and host-staged\n"
        "                          copies passed the self-test; tcp refuses\n"
        "                          them so that the engines use their base\n"
        "                          transport (also the outcome of any\n"
        "                          host-path error)\n"
        "  --host-staging-mb <MB>  pinned host memory for host-staged copies,\n"
        "                          split over the NUMA nodes with GPUs and\n"
        "                          reserved at start (default 128; counts\n"
        "                          against the container's memory limit)\n"
        "  --host-staging-wait-ms <ms>  longest wait of a copy for staging\n"
        "                          slots before it is refused (default 100)\n"
        "  --deny-peer <bus id>    deny every pair with this GPU (PCI bus id,\n"
        "                          e.g. 0000:3b:00.0); repeatable\n"
        "  --deny-pair <a>,<b>     deny the pair of these two GPUs; "
        "repeatable\n"
        "                          Copies between the GPUs of a denied pair\n"
        "                          are refused (no P2P) and the engines use\n"
        "                          their base transport for them.\n"
        "  --eager-init            create a CUDA context on every visible GPU\n"
        "                          and enable peer access for all pairs at\n"
        "                          start; by default a GPU is initialized "
        "when\n"
        "                          the first block on it is registered, so\n"
        "                          GPUs without Mooncake engines keep no\n"
        "                          daemon context after the start-up test\n"
        "                          (CUDA_VISIBLE_DEVICES also limits the set)\n"
        "  -h, --help              show this help\n"
        "\n"
        "Counters (log line / --stats): active_clients, registrations,\n"
        "copy_requests, copy_entries, copy_bytes, copy_failures, avg/max\n"
        "copy latency in microseconds (split into avg_plan_us and\n"
        "avg_exec_us), coalesced_entries, kernel_entries/bytes and\n"
        "ce_entries/bytes (copy engines), handle_opens, p2p_selftest,\n"
        "denied_pairs (denied ordered src>dst directions: count and bus\n"
        "ids), denied_copy_requests, untested_pairs and\n"
        "untested_host_paths (tests that could not run, waiting for a\n"
        "retry), staged_copies, staged_bytes, staged_failures (host-staged\n"
        "copies).\n",
        argv0);
}

}  // namespace

int main(int argc, char **argv) {
    std::string socket_path;
    mode_t socket_mode = 0666;
    int stats_interval = 60;
    bool do_stats = false, do_ping = false, eager = false;
    int drain_timeout_ms = 10000;
    std::vector<std::string> deny_peers, deny_pairs;
    bool startup_selftest = true;
    bool selftest_child = false;
    std::string selftest_pairs, selftest_stage_pairs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a value\n", name);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--socket") {
            socket_path = need("--socket");
        } else if (a.rfind("--socket=", 0) == 0) {
            socket_path = a.substr(9);
        } else if (a == "--socket-mode") {
            socket_mode =
                static_cast<mode_t>(strtol(need("--socket-mode"), nullptr, 8));
        } else if (a == "--stats-interval") {
            stats_interval = atoi(need("--stats-interval"));
        } else if (a == "--stats") {
            do_stats = true;
        } else if (a == "--ping") {
            do_ping = true;
        } else if (a == "--eager-init") {
            eager = true;
        } else if (a == "--gather-threshold") {
            g_cfg.gather_threshold =
                strtoull(need("--gather-threshold"), nullptr, 0);
        } else if (a == "--gather-on") {
            std::string v = need("--gather-on");
            if (v != "dst" && v != "src") {
                fprintf(stderr, "--gather-on must be dst or src\n");
                return 2;
            }
            g_cfg.gather_on_src = (v == "src");
        } else if (a == "--no-coalesce") {
            g_cfg.coalesce = false;
        } else if (a == "--no-startup-selftest") {
            startup_selftest = false;
        } else if (a == "--p2p-fallback" ||
                   a.rfind("--p2p-fallback=", 0) == 0) {
            const std::string v =
                a == "--p2p-fallback" ? need("--p2p-fallback") : a.substr(15);
            if (v != "host" && v != "tcp") {
                fprintf(stderr, "--p2p-fallback must be host or tcp\n");
                return 2;
            }
            g_fallback_host = v == "host";
        } else if (a == "--host-staging-mb") {
            const long mb = atol(need("--host-staging-mb"));
            if (mb < 2) {
                fprintf(stderr, "--host-staging-mb must be at least 2\n");
                return 2;
            }
            g_host_staging_bytes = uint64_t(mb) << 20;
        } else if (a == "--p2p-selftest-timeout") {
            g_selftest_timeout_s = atoi(need("--p2p-selftest-timeout"));
            if (g_selftest_timeout_s <= 0) {
                fprintf(stderr, "--p2p-selftest-timeout must be > 0\n");
                return 2;
            }
        } else if (a == "--p2p-selftest-child") {  // internal
            selftest_child = true;
        } else if (a == "--p2p-selftest-pairs") {  // internal
            selftest_pairs = need("--p2p-selftest-pairs");
        } else if (a == "--p2p-selftest-stage-pairs") {  // internal
            selftest_stage_pairs = need("--p2p-selftest-stage-pairs");
        } else if (a == "--host-staging-wait-ms") {
            g_host_staging_wait_ms = atoi(need("--host-staging-wait-ms"));
            if (g_host_staging_wait_ms < 0) {
                fprintf(stderr, "--host-staging-wait-ms must be >= 0\n");
                return 2;
            }
        } else if (a == "--p2p-selftest" ||
                   a.rfind("--p2p-selftest=", 0) == 0) {
            std::string v =
                a == "--p2p-selftest" ? need("--p2p-selftest") : a.substr(15);
            if (!parseSelfTestMode(v, g_selftest)) {
                fprintf(stderr,
                        "--p2p-selftest must be enforce, warn or off\n");
                return 2;
            }
        } else if (a == "--deny-peer" || a.rfind("--deny-peer=", 0) == 0) {
            std::string v =
                a == "--deny-peer" ? need("--deny-peer") : a.substr(12);
            PciAddr addr;
            if (!parsePciAddr(v, addr)) {
                fprintf(stderr,
                        "--deny-peer: %s is not a PCI bus id "
                        "([domain:]bus:device[.function])\n",
                        v.c_str());
                return 2;
            }
            deny_peers.push_back(v);
        } else if (a == "--deny-pair" || a.rfind("--deny-pair=", 0) == 0) {
            std::string v =
                a == "--deny-pair" ? need("--deny-pair") : a.substr(12);
            const size_t comma = v.find(',');
            PciAddr x, y;
            if (comma == std::string::npos ||
                !parsePciAddr(v.substr(0, comma), x) ||
                !parsePciAddr(v.substr(comma + 1), y)) {
                fprintf(stderr,
                        "--deny-pair: expected <bus id>,<bus id>, got %s\n",
                        v.c_str());
                return 2;
            }
            deny_pairs.push_back(v);
        } else if (a == "--drain-timeout-ms") {
            drain_timeout_ms = atoi(need("--drain-timeout-ms"));
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }
    if (selftest_child) {
        // A fresh process (posix_spawn of this binary): CUDA initializes
        // here for the first time.
        const int rc = runSelfTestChild(selftest_pairs, selftest_stage_pairs);
        fflush(stdout);
        fflush(stderr);
        _exit(rc);
    }
    {
        char exe[4096];
        const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
        g_self_exe = n > 0 ? std::string(exe, size_t(n)) : std::string(argv[0]);
    }
    if (socket_path.empty()) {
        const char *env = getenv("MC_NVLINK_PROXY_SOCKET");
        if (env) socket_path = env;
    }
    if (socket_path.empty()) {
        usage(argv[0]);
        return 2;
    }
    if (do_stats) return queryDaemon(socket_path, MsgType::kStats);
    if (do_ping) return queryDaemon(socket_path, MsgType::kPing);

    signal(SIGPIPE, SIG_IGN);
    g_start_us = nowUs();
    {
        std::random_device rd;
        g_epoch = (uint64_t(rd()) << 32) ^ rd() ^ uint64_t(getpid()) ^
                  uint64_t(time(nullptr));
        if (g_epoch == 0) g_epoch = 1;
    }
    LOGI("starting, protocol v%u, epoch %016" PRIx64
         ", gather kernel for copies < %" PRIu64
         " bytes on the %s GPU, "
         "coalescing %s",
         kProtocolVersion, g_epoch, g_cfg.gather_threshold,
         g_cfg.gather_on_src ? "sending" : "receiving",
         g_cfg.coalesce ? "on" : "off");
    if (!initDevices()) return 1;
    // Command-line denials, by PCI bus id (a DaemonSet may name GPUs that
    // some nodes do not have: those are ignored with a warning).
    auto findBus = [](const std::string &v) -> int {
        PciAddr addr;
        if (!parsePciAddr(v, addr)) return -1;
        for (const Device &d : g_devices)
            if (d.addr == addr) return d.ordinal;
        return -1;
    };
    for (const std::string &v : deny_peers) {
        const int d = findBus(v);
        if (d < 0) {
            LOGW("--deny-peer %s: no such GPU visible to the daemon; ignored",
                 v.c_str());
            continue;
        }
        g_policy.denyDevice(d, PeerPolicy::kStatic);
        LOGI(
            "--deny-peer %s: P2P between gpu %d (%s) and every other GPU is "
            "denied",
            v.c_str(), d, g_devices[d].pci.c_str());
    }
    for (const std::string &v : deny_pairs) {
        const size_t comma = v.find(',');
        const int x = findBus(v.substr(0, comma));
        const int y = findBus(v.substr(comma + 1));
        if (x < 0 || y < 0 || x == y) {
            LOGW(
                "--deny-pair %s: not two different GPUs visible to the "
                "daemon; ignored",
                v.c_str());
            continue;
        }
        g_policy.denyPair(x, y, PeerPolicy::kStatic);
        LOGI(
            "--deny-pair %s: P2P between gpu %d (%s) and gpu %d (%s) is "
            "denied",
            v.c_str(), x, g_devices[x].pci.c_str(), y,
            g_devices[y].pci.c_str());
    }
    // Host staging: the pool must hold two slots per NUMA node with GPUs.
    if (g_fallback_host) {
        uint64_t slot = 0;
        size_t slots = 0;
        const size_t nodes = gpuNumaNodes();
        if (!stagingGeometry(g_host_staging_bytes, nodes, &slot, &slots)) {
            LOGE(
                "host staging disabled: --host-staging-mb %" PRIu64
                " leaves fewer than two 1 MiB slots per NUMA node (%zu node(s) "
                "with GPUs; at least %zu MiB needed); copies that cannot use "
                "P2P use the engines' base transport",
                g_host_staging_bytes >> 20, nodes, 2 * nodes);
            g_fallback_host = false;
        } else if (g_selftest != SelfTestMode::kOff) {
            LOGI("host staging: %zu x %" PRIu64
                 " MiB pinned slots per NUMA node, %zu node(s), wait for "
                 "slots at most %d ms",
                 slots, slot >> 20, nodes, g_host_staging_wait_ms);
            // Reserve (and touch) the pools now, in the background: a
            // memory limit too small for them shows at start, not mid-serve.
            std::set<int> numa;
            for (const Device &d : g_devices) numa.insert(d.numa);
            for (int node : numa) {
                try {
                    std::thread([node]() {
                        std::string err;
                        if (!stagingPool(node).reserve(&err))
                            LOGE("host staging pool on NUMA node %d: %s", node,
                                 err.c_str());
                    }).detach();
                } catch (const std::system_error &e) {
                    LOGE(
                        "cannot start a thread to reserve the staging pool: "
                        "%s",
                        e.what());
                }
            }
        }
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (socket_path.size() >= sizeof(addr.sun_path)) {
        LOGE("socket path too long: %s", socket_path.c_str());
        return 2;
    }
    memcpy(addr.sun_path, socket_path.c_str(), socket_path.size() + 1);
    struct stat st;
    if (lstat(socket_path.c_str(), &st) == 0) {
        if (!S_ISSOCK(st.st_mode)) {
            LOGE("%s exists and is not a socket", socket_path.c_str());
            return 2;
        }
        int probe = connectTo(socket_path);
        if (probe >= 0) {
            close(probe);
            LOGE("another daemon is already listening on %s",
                 socket_path.c_str());
            return 2;
        }
        unlink(socket_path.c_str());  // stale socket of a previous run
    }
    if (eager) {
        for (size_t i = 0; i < g_devices.size(); ++i) {
            if (!ensureDevice(static_cast<int>(i))) return 1;
        }
    }
    int lfd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (lfd < 0 ||
        bind(lfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
        listen(lfd, 128) != 0) {
        LOGE("cannot listen on %s: %s", socket_path.c_str(), strerror(errno));
        return 1;
    }
    chmod(socket_path.c_str(), socket_mode);
    g_socket_path = socket_path;
    struct sigaction sa{};
    sa.sa_handler = onSignal;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);
    LOGI("listening on %s", socket_path.c_str());

    // P2P self-test, after the socket is up: engines connect and register
    // meanwhile; in enforce mode pairs not tested yet are refused (base
    // transport) until their test passes.
    g_tester.start();
    if (g_selftest == SelfTestMode::kOff) {
        LOGW(
            "P2P self-test off (--p2p-selftest=off): peer copies are trusted "
            "without verification; host paths are not verified either, so "
            "pairs denied on the command line and pairs without P2P use the "
            "engines' base transport (TCP)");
    } else if (startup_selftest) {
        std::vector<std::pair<int, int>> pairs;
        const int n = static_cast<int>(g_devices.size());
        for (int x = 0; x < n; ++x) {
            if (g_devices[x].exclusive) continue;
            pairs.push_back({x, x});  // host path
            for (int y = x + 1; y < n; ++y)
                if (!g_devices[y].exclusive) pairs.push_back({x, y});
        }
        LOGI(
            "P2P self-test (%s): testing GPU pairs and host paths in a child "
            "process (timeout %d s); refused pairs %s",
            selfTestModeName(g_selftest), g_selftest_timeout_s,
            g_fallback_host ? "are host-staged when both host paths pass"
                            : "use TCP (--p2p-fallback=tcp)");
        g_tester.request(pairs);
    } else {
        LOGI(
            "P2P self-test (%s): pairs are tested in a child process when "
            "engines first register on both GPUs (--no-startup-selftest); "
            "refused pairs %s",
            selfTestModeName(g_selftest),
            g_fallback_host ? "are host-staged when both host paths pass"
                            : "use TCP (--p2p-fallback=tcp)");
    }

    std::thread stats_thread([stats_interval]() {
        if (stats_interval <= 0) return;
        std::string last;
        uint64_t next = nowUs() + uint64_t(stats_interval) * 1000000;
        while (!g_stop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (nowUs() < next) continue;
            next = nowUs() + uint64_t(stats_interval) * 1000000;
            std::string s = statsText();
            // Skip the line when nothing but uptime changed.
            std::string key = s.substr(s.find(" devices="));
            if (key != last) LOGI("stats %s", s.c_str());
            last = key;
        }
    });

    while (!g_stop) {
        pollfd p{lfd, POLLIN, 0};
        int rc = poll(&p, 1, 500);
        if (rc <= 0) continue;
        int fd = accept4(lfd, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd < 0) continue;
        std::thread(handleConnection, fd).detach();
    }
    // Stop accepting, then let copies already running finish and reply so
    // their clients do not have to retry them.
    g_tester.stop();
    close(lfd);
    unlink(g_socket_path.c_str());
    const uint64_t drain_until = nowUs() + uint64_t(drain_timeout_ms) * 1000;
    int inflight = g_inflight_copies.load();
    if (inflight > 0)
        LOGI("draining %d in-flight copy request(s) (up to %d ms)", inflight,
             drain_timeout_ms);
    while ((inflight = g_inflight_copies.load()) > 0 && nowUs() < drain_until)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (inflight > 0)
        LOGW("exiting with %d copy request(s) still running", inflight);
    LOGI("shutting down: %s", statsText().c_str());
    stats_thread.join();
    // Exit without tearing down the CUDA contexts under in-flight copies.
    _exit(0);
}
