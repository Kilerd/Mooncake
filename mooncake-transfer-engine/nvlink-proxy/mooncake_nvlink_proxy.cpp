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

#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/un.h>
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

#include "transport/nvlink_proxy_transport/nvlink_proxy_protocol.h"

using namespace mooncake::nvlink_proxy;

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
    char buf[1024];
    snprintf(
        buf, sizeof(buf),
        "epoch=%016" PRIx64 " uptime_s=%" PRIu64
        " devices=%zu active_clients=%" PRIu64 " active_connections=%" PRIu64
        " registrations=%zu copy_requests=%" PRIu64 " copy_entries=%" PRIu64
        " copy_bytes=%" PRIu64 " copy_failures=%" PRIu64 " avg_copy_us=%" PRIu64
        " max_copy_us=%" PRIu64 " handle_opens=%" PRIu64,
        g_epoch, (nowUs() - g_start_us) / 1000000, g_devices.size(),
        g_stats.active_clients.load(), g_stats.active_connections.load(), regs,
        reqs, g_stats.copy_entries.load(), g_stats.copy_bytes.load(),
        g_stats.copy_failures.load(), reqs ? us / reqs : 0,
        g_stats.copy_us_max.load(), g_stats.handle_opens.load());
    return buf;
}

// ------------------------------------------------------------ copy engine
// Per connection-thread streams, one per pulling device.
struct ThreadStreams {
    std::unordered_map<int, cudaStream_t> streams;
    ~ThreadStreams() {
        for (auto &kv : streams) {
            cudaSetDevice(kv.first);
            cudaStreamDestroy(kv.second);
        }
    }
    cudaError_t get(int dev, cudaStream_t *out) {
        auto it = streams.find(dev);
        if (it != streams.end()) {
            *out = it->second;
            return cudaSuccess;
        }
        cudaError_t e = cudaSetDevice(dev);
        if (e != cudaSuccess) return e;
        cudaStream_t s;
        e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        if (e != cudaSuccess) return e;
        streams[dev] = s;
        *out = s;
        return cudaSuccess;
    }
};

thread_local ThreadStreams t_streams;
std::atomic<bool> g_batch_api_usable{true};

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

struct CopyResult {
    ProxyStatus status = ProxyStatus::kOk;
    uint32_t failed_index = 0;
    uint64_t bytes = 0;
    std::string message;
};

std::shared_ptr<Registration> lookupReg(uint64_t client, uint64_t base) {
    std::lock_guard<std::mutex> lock(g_reg_mu);
    auto it = g_regs.find({client, base});
    return it == g_regs.end() ? nullptr : it->second;
}

CopyResult doCopy(const CopyEntry *entries, uint32_t count, uint32_t budget_ms,
                  uint64_t t_recv_us) {
    CopyResult r;
    r.failed_index = count;
    struct Planned {
        int dev;
        void *dst;
        void *src;
        size_t len;
    };
    std::vector<Planned> plan;
    plan.reserve(count);
    // Keep every registration referenced by this request alive until the
    // copies completed, even if its owner disconnects meanwhile.
    std::vector<std::shared_ptr<Registration>> keep;
    std::shared_ptr<Registration> last_src, last_dst;
    auto fail = [&](ProxyStatus st, uint32_t idx, std::string msg) {
        r.status = st;
        r.failed_index = idx;
        r.message = std::move(msg);
        return r;
    };
    for (uint32_t i = 0; i < count; ++i) {
        const CopyEntry &e = entries[i];
        if (e.length == 0) continue;
        if (!last_src || last_src->client_id != e.src_client ||
            last_src->base != e.src_base) {
            last_src = lookupReg(e.src_client, e.src_base);
            if (!last_src) {
                char m[128];
                snprintf(m, sizeof(m),
                         "source block client=%016" PRIx64 " base=%#" PRIx64
                         " is not registered",
                         e.src_client, e.src_base);
                return fail(ProxyStatus::kUnknownSource, i, m);
            }
            keep.push_back(last_src);
        }
        if (!last_dst || last_dst->client_id != e.dst_client ||
            last_dst->base != e.dst_base) {
            last_dst = lookupReg(e.dst_client, e.dst_base);
            if (!last_dst) {
                char m[128];
                snprintf(m, sizeof(m),
                         "destination block client=%016" PRIx64
                         " base=%#" PRIx64 " is not registered",
                         e.dst_client, e.dst_base);
                return fail(ProxyStatus::kUnknownDestination, i, m);
            }
            keep.push_back(last_dst);
        }
        if (e.src_offset > last_src->size ||
            e.length > last_src->size - e.src_offset ||
            e.dst_offset > last_dst->size ||
            e.length > last_dst->size - e.dst_offset) {
            return fail(ProxyStatus::kOutOfRange, i,
                        "copy range exceeds a registered block");
        }
        const int dev = last_dst->device;
        if (last_src->device != dev && !g_peer[dev][last_src->device].load()) {
            char m[96];
            snprintf(m, sizeof(m), "gpu %d cannot access gpu %d (no P2P)", dev,
                     last_src->device);
            return fail(ProxyStatus::kNoPeerAccess, i, m);
        }
        char *src_map = nullptr, *dst_map = nullptr;
        bool src_new = false, dst_new = false;
        cudaError_t ce = last_src->mapFor(dev, &src_map, &src_new);
        if (ce == cudaSuccess) ce = last_dst->mapFor(dev, &dst_map, &dst_new);
        if (src_new) g_stats.handle_opens++;
        if (dst_new) g_stats.handle_opens++;
        if (ce != cudaSuccess) {
            if (isStickyCudaError(ce)) dieOnStickyError(ce, "IPC open");
            cudaGetLastError();
            return fail(
                ProxyStatus::kCudaError, i,
                std::string("cudaIpcOpenMemHandle: ") + cudaGetErrorString(ce));
        }
        plan.push_back(
            {dev, dst_map + e.dst_offset, src_map + e.src_offset, e.length});
        r.bytes += e.length;
    }

    if (budget_ms && nowUs() - t_recv_us > uint64_t(budget_ms) * 1000) {
        return fail(ProxyStatus::kDeadlineExceeded, count,
                    "start budget exceeded before the copy was issued");
    }

    // Group by pulling device (normally a single device per request).
    std::map<int, std::vector<size_t>> by_dev;
    for (size_t i = 0; i < plan.size(); ++i) by_dev[plan[i].dev].push_back(i);
    std::vector<std::pair<int, cudaStream_t>> used;
    for (auto &kv : by_dev) {
        const int dev = kv.first;
        cudaStream_t stream;
        cudaError_t ce = t_streams.get(dev, &stream);
        if (ce == cudaSuccess) ce = cudaSetDevice(dev);
        std::vector<void *> dsts, srcs;
        std::vector<size_t> sizes;
        dsts.reserve(kv.second.size());
        srcs.reserve(kv.second.size());
        sizes.reserve(kv.second.size());
        for (size_t i : kv.second) {
            dsts.push_back(plan[i].dst);
            srcs.push_back(plan[i].src);
            sizes.push_back(plan[i].len);
        }
        if (ce == cudaSuccess) ce = issueCopies(dsts, srcs, sizes, stream);
        if (ce != cudaSuccess) {
            if (isStickyCudaError(ce)) dieOnStickyError(ce, "copy submit");
            cudaGetLastError();
            // Drain what was already queued before replying.
            for (auto &u : used) cudaStreamSynchronize(u.second);
            cudaStreamSynchronize(stream);
            return fail(ProxyStatus::kCudaError, count,
                        std::string("copy submit: ") + cudaGetErrorString(ce));
        }
        used.emplace_back(dev, stream);
    }
    cudaError_t first_err = cudaSuccess;
    for (auto &u : used) {
        cudaError_t ce = cudaStreamSynchronize(u.second);
        if (ce != cudaSuccess && first_err == cudaSuccess) first_err = ce;
    }
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
                } else {
                    g_stats.copy_failures++;
                    out += res.message;
                    LOGW("copy from client %016" PRIx64
                         " failed: %s (%s, "
                         "entry %u of %u)",
                         c.client_id, proxyStatusName(res.status),
                         res.message.c_str(), res.failed_index, rh.count);
                }
                ok = sendMsg(fd, type, h.seq, res.status, out.data(),
                             out.size());
                break;
            }
            case MsgType::kPing: {
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
bool initDevices() {
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
        g_devices.push_back(d);
        const bool exclusive = prop.computeMode != cudaComputeModeDefault;
        LOGI("gpu %d: %s %s pci %s%s", i, d.name.c_str(),
             hexUuid(d.uuid).c_str(), d.pci.c_str(),
             exclusive ? " compute mode NOT DEFAULT" : "");
        if (exclusive) {
            LOGW(
                "gpu %d is in an exclusive/prohibited compute mode; the "
                "daemon cannot create a context next to an engine using it, "
                "so copies involving it will be refused (engines fall back "
                "to their base transport). The proxy needs compute mode "
                "DEFAULT.",
                i);
        }
    }
    for (int a = 0; a < kMaxDevices; ++a) {
        g_dev_ready[a] = false;
        for (int b = 0; b < kMaxDevices; ++b) g_peer[a][b] = false;
    }
    return n > 0;
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

std::atomic<bool> g_stop{false};
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
        "  --eager-init            create a CUDA context on every visible GPU\n"
        "                          and enable peer access for all pairs at\n"
        "                          start; by default a GPU is initialized "
        "when\n"
        "                          the first block on it is registered, so\n"
        "                          GPUs without Mooncake engines are "
        "untouched\n"
        "                          (CUDA_VISIBLE_DEVICES also limits the set)\n"
        "  -h, --help              show this help\n"
        "\n"
        "Counters (log line / --stats): active_clients, registrations,\n"
        "copy_requests, copy_entries, copy_bytes, copy_failures, avg/max\n"
        "copy latency in microseconds, handle_opens.\n",
        argv0);
}

}  // namespace

int main(int argc, char **argv) {
    std::string socket_path;
    mode_t socket_mode = 0666;
    int stats_interval = 60;
    bool do_stats = false, do_ping = false, eager = false;
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
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
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
    LOGI("starting, protocol v%u, epoch %016" PRIx64, kProtocolVersion,
         g_epoch);
    if (!initDevices()) return 1;
    if (eager) {
        for (size_t i = 0; i < g_devices.size(); ++i) {
            if (!ensureDevice(static_cast<int>(i))) return 1;
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
    LOGI("shutting down: %s", statsText().c_str());
    close(lfd);
    unlink(g_socket_path.c_str());
    stats_thread.join();
    // Exit without tearing down the CUDA contexts under in-flight copies.
    _exit(0);
}
