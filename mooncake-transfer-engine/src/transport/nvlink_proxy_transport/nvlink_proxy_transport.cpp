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

#include "transport/nvlink_proxy_transport/nvlink_proxy_transport.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <thread>

#include "common.h"
#include "error.h"
#include "fresh_random.h"

namespace mooncake {

using namespace nvlink_proxy;

namespace {

constexpr const char *kProtocol = "nvlink_proxy";
constexpr size_t kMaxEntriesPerCopy = 65536;
constexpr int64_t kFallbackLogIntervalNs = 5LL * 1000 * 1000 * 1000;

int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

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

// writeAll() of |len| bytes that carries |pass_fd| (SCM_RIGHTS) with the
// first of them.
bool sendWithFd(int fd, const void *buf, size_t len, int pass_fd) {
    iovec iov{const_cast<void *>(buf), len};
    alignas(cmsghdr) char ctrl[CMSG_SPACE(sizeof(int))] = {};
    msghdr m{};
    m.msg_iov = &iov;
    m.msg_iovlen = 1;
    m.msg_control = ctrl;
    m.msg_controllen = sizeof(ctrl);
    cmsghdr *c = CMSG_FIRSTHDR(&m);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(c), &pass_fd, sizeof(int));
    ssize_t n;
    do n = sendmsg(fd, &m, MSG_NOSIGNAL);
    while (n < 0 && errno == EINTR);
    if (n < 0) return false;
    return static_cast<size_t>(n) == len ||
           writeAll(fd, static_cast<const char *>(buf) + n, len - n);
}

bool readAll(int fd, void *buf, size_t len) {
    char *p = static_cast<char *>(buf);
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n == 0) {
            errno = ECONNRESET;
            return false;
        }
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

int envInt(const char *name, int def) {
    const char *v = getenv(name);
    if (!v || !*v) return def;
    char *end = nullptr;
    long x = strtol(v, &end, 10);
    if (!end || *end != '\0') {
        LOG(WARNING) << "nvlink_proxy: ignoring invalid " << name << "=" << v;
        return def;
    }
    return static_cast<int>(x);
}

// Locality key: MC_NODE_ID, else NODE_NAME (Kubernetes downward API), else
// the hostname. Two engines use the proxy only when their keys match.
std::string resolveNodeId() {
    std::string id;
    for (const char *name : {"MC_NODE_ID", "NODE_NAME"}) {
        const char *v = getenv(name);
        if (v && *v) {
            id = v;
            break;
        }
    }
    if (id.empty()) {
        char host[256] = {};
        if (gethostname(host, sizeof(host) - 1) == 0) id = host;
    }
    if (id.empty()) id = "unknown";
    for (auto &c : id) {
        if (c == '|' || std::isspace(static_cast<unsigned char>(c))) c = '_';
    }
    return id;
}

uint64_t nonZeroRandom() {
    uint64_t v;
    do v = freshRandom64();
    while (v == 0);
    return v;
}

// The process token of HELLO: the engines of a process share it while any
// of them is installed, and one installed when none is draws a new one.
// Processes restored from one snapshot (same memory, same pid) therefore
// report different tokens once their engines were shut down before the
// checkpoint and recreated after the restore.
std::mutex g_token_mu;
uint64_t g_token = 0;
size_t g_token_users = 0;

uint64_t acquireProcessToken() {
    std::lock_guard<std::mutex> lock(g_token_mu);
    if (g_token_users++ == 0) g_token = nonZeroRandom();
    return g_token;
}

void releaseProcessToken() {
    std::lock_guard<std::mutex> lock(g_token_mu);
    if (g_token_users > 0) --g_token_users;
}

// A POSIX file descriptor of the allocation mapped at |base| (REGISTER_FD);
// -1 when it cannot be exported. Exported anew at every registration, so
// that the engine keeps no such descriptor open.
int exportVmmFd(uint64_t base, CUresult *error) {
    CUmemGenericAllocationHandle h;
    *error = cuMemRetainAllocationHandle(&h, reinterpret_cast<void *>(base));
    if (*error != CUDA_SUCCESS) return -1;
    int fd = -1;
    *error = cuMemExportToShareableHandle(
        &fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
    cuMemRelease(h);
    if (*error != CUDA_SUCCESS) return -1;
    fcntl(fd, F_SETFD, FD_CLOEXEC);  // no child keeps the allocation alive
    return fd;
}

// Makes GPU |dev|'s primary context current until it goes out of scope,
// then restores the caller's. cuMemGetAddressRange answers in the current
// context only, and a process may register buffers of several GPUs from
// one thread (or from the health thread, which has no context). Does
// nothing when a context of |dev| is current already.
class DeviceScope {
   public:
    explicit DeviceScope(int dev) {
        result_ = cuDeviceGet(&dev_, dev);
        if (result_ != CUDA_SUCCESS) return;
        CUcontext cur = nullptr;
        CUdevice cur_dev = 0;
        if (cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur &&
            cuCtxGetDevice(&cur_dev) == CUDA_SUCCESS && cur_dev == dev_)
            return;
        result_ = cuDevicePrimaryCtxRetain(&ctx_, dev_);
        if (result_ != CUDA_SUCCESS) return;
        retained_ = true;
        result_ = cuCtxPushCurrent(ctx_);
        pushed_ = result_ == CUDA_SUCCESS;
    }
    ~DeviceScope() {
        CUcontext popped;
        if (pushed_) cuCtxPopCurrent(&popped);
        if (retained_) cuDevicePrimaryCtxRelease(dev_);
    }
    DeviceScope(const DeviceScope &) = delete;
    DeviceScope &operator=(const DeviceScope &) = delete;

    // CUDA_SUCCESS when a context of the GPU is current.
    CUresult result() const { return result_; }

   private:
    CUdevice dev_ = 0;
    CUcontext ctx_ = nullptr;
    CUresult result_ = CUDA_SUCCESS;
    bool retained_ = false;
    bool pushed_ = false;
};

const char *reasonName(int reason) {
    switch (reason) {
        case 0:
            return "daemon unavailable";
        case 1:
            return "daemon error";
        case 3:
            return "GPU pair still under the daemon's P2P self-test";
        default:
            return "request not servable by the proxy";
    }
}

}  // namespace

NvlinkProxyTransport::NvlinkProxyTransport() = default;

NvlinkProxyTransport::~NvlinkProxyTransport() {
    stop(false);
    // stop() may have run before (shutdown()); nothing reopens a connection
    // after it, these only close what a misuse could have left.
    {
        std::lock_guard<std::mutex> lock(control_mu_);
        closeControlLocked();
    }
    std::lock_guard<std::mutex> lock(pool_mu_);
    for (auto &c : idle_) close(c.fd);
    idle_.clear();
}

int NvlinkProxyTransport::shutdown() { return stop(true) ? 0 : -1; }

bool NvlinkProxyTransport::stop(bool release) {
    std::lock_guard<std::mutex> stop_lock(stop_mu_);
    if (stopped_) return release_confirmed_;
    stopped_ = true;
    {
        std::lock_guard<std::mutex> lock(health_mu_);
        stop_ = true;
    }
    health_cv_.notify_all();
    {
        std::lock_guard<std::mutex> lock(state_mu_);
        shut_down_ = true;
    }
    state_cv_.notify_all();
    if (health_thread_.joinable()) health_thread_.join();
    maybeLogStats(true);
    bool confirmed = true;
    {
        std::lock_guard<std::mutex> lock(control_mu_);
        if (release) confirmed = releaseLocked();
        closeControlLocked();
    }
    {
        std::lock_guard<std::mutex> lock(pool_mu_);
        for (auto &c : idle_) close(c.fd);
        idle_.clear();
    }
    {
        // Unconfirmed attempts stay counted (their batches never quiesce);
        // abandoned fallback batches are leaked: the base transport may
        // still reference them.
        std::lock_guard<std::mutex> lock(unconfirmed_mu_);
        for (auto &u : unconfirmed_) {
            if (u.fd >= 0) close(u.fd);
            u.fd = -1;
        }
    }
    if (process_token_) releaseProcessToken();
    release_confirmed_ = confirmed;
    return confirmed;
}

// Requires control_mu_.
bool NvlinkProxyTransport::releaseLocked() {
    // No block was ever sent to a daemon: none can have mapped this memory.
    if (!ever_registered_) return true;
    char id[32];
    snprintf(id, sizeof(id), "%016" PRIx64, client_id_);
    // A daemon this engine lost after registering may still map its memory
    // until the copies in flight that use it drained (each within its
    // request budget, the daemon's drain within 10 s by default): neither
    // its absence nor another daemon's answer proves it gone before that.
    const int64_t lost_ago_ms = lost_ns_ ? (nowNs() - lost_ns_) / 1000000 : -1;
    const bool lost_recently =
        lost_ns_ != 0 && lost_ago_ms < std::max(timeout_ms_, 10000);
    int fd = control_fd_;
    uint64_t epoch = epoch_.load();
    const bool own = fd < 0;
    if (own) {
        // The control connection is gone (the daemon drops what it
        // registered by itself); one opened for RELEASE waits for the
        // mappings of that memory to be closed.
        fd = connectSocket(2000);
        if (fd < 0) {
            const int err = errno;
            if ((err == ENOENT || err == ECONNREFUSED) && !lost_recently) {
                LOG(WARNING) << "nvlink_proxy: client " << id
                             << ": no daemon listens at " << socket_path_
                             << " (" << strerror(err)
                             << "); a daemon that exited holds no mapping "
                                "of this engine's memory";
                return true;
            }
            LOG(ERROR) << "nvlink_proxy: client " << id
                       << ": cannot reach the daemon at " << socket_path_
                       << " to release this engine's memory (" << strerror(err)
                       << ")"
                       << (lost_recently
                               ? "; the daemon lost " +
                                     std::to_string(lost_ago_ms) +
                                     " ms ago may still be draining copies "
                                     "that use it"
                               : std::string());
            return false;
        }
        if (!hello(fd, true, epoch)) {
            const int err = errno;
            close(fd);
            LOG(ERROR) << "nvlink_proxy: client " << id
                       << ": HELLO to release this engine's memory failed ("
                       << strerror(err) << ")";
            return false;
        }
    }
    // The daemon answers once the mappings are closed: wait longer than it.
    const int wait_ms = release_wait_ms_ + 2000;
    struct timeval tv;
    tv.tv_sec = wait_ms / 1000;
    tv.tv_usec = (wait_ms % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ReleaseReq req{};
    req.wait_ms = static_cast<uint32_t>(release_wait_ms_);
    ProxyStatus st = ProxyStatus::kOk;
    std::string reply;
    const int64_t t0 = nowNs();
    const bool answered =
        rpc(fd, MsgType::kRelease, &req, sizeof(req), st, reply);
    const int err = errno;
    if (own) close(fd);
    const int64_t ms = (nowNs() - t0) / 1000000;
    ReleaseResp resp{};
    const bool sized = answered && reply.size() == sizeof(resp);
    if (sized) memcpy(&resp, reply.data(), sizeof(resp));
    if (sized && st == ProxyStatus::kOk) {
        if (lost_recently && lost_epoch_ != epoch) {
            LOG(ERROR) << "nvlink_proxy: client " << id
                       << ": the daemon holds no mapping of this engine's "
                          "memory, but the previous daemon, lost "
                       << lost_ago_ms
                       << " ms ago, may still be draining copies that use "
                          "it; unconfirmed";
            return false;
        }
        LOG(INFO) << "nvlink_proxy: client " << id << " released "
                  << resp.dropped
                  << " registration(s); the daemon holds no IPC mapping of "
                     "this engine's memory ("
                  << ms << " ms)";
        return true;
    }
    if (!answered) {
        LOG(ERROR) << "nvlink_proxy: client " << id
                   << ": no answer to RELEASE (" << strerror(err)
                   << "); the daemon drops this engine's registrations once "
                      "it is disconnected and closes their mappings when "
                      "the copies using them end, unconfirmed";
    } else if (sized && st == ProxyStatus::kDeadlineExceeded) {
        LOG(ERROR) << "nvlink_proxy: client " << id << ": " << resp.mappings
                   << " IPC mapping(s) of this engine's memory still used by "
                      "copies in flight after "
                   << ms << " ms (MC_NVLINK_PROXY_RELEASE_WAIT_MS="
                   << release_wait_ms_ << ")";
    } else {
        LOG(ERROR) << "nvlink_proxy: client " << id
                   << ": the daemon refused RELEASE: " << proxyStatusName(st)
                   << (sized ? "" : " " + reply)
                   << (st == ProxyStatus::kBadRequest
                           ? " (a daemon older than this engine: it closes "
                             "the mappings of this engine's memory after "
                             "the disconnect, unconfirmed)"
                           : "");
    }
    return false;
}

bool NvlinkProxyTransport::vmmBlock(Block &block, bool *is_vmm,
                                    std::string &why) {
    *is_vmm = false;
    CUmemGenericAllocationHandle h;
    if (cuMemRetainAllocationHandle(&h, reinterpret_cast<void *>(block.base)) !=
        CUDA_SUCCESS) {
        why = "not virtual memory management memory";
        return false;
    }
    *is_vmm = true;
    CUmemAllocationProp prop{};
    const CUresult r = cuMemGetAllocationPropertiesFromHandle(&prop, h);
    cuMemRelease(h);
    if (r != CUDA_SUCCESS) {
        why = "cuMemGetAllocationPropertiesFromHandle: CUresult " +
              std::to_string(static_cast<int>(r));
        return false;
    }
    if (!(prop.requestedHandleTypes &
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)) {
        why =
            "allocated without CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR in "
            "requestedHandleTypes";
        return false;
    }
    // One allocation from the first byte of the block to its last: the
    // daemon maps [0, size) of the allocation it imports.
    unsigned long long first = 0, last = 0;
    if (cuPointerGetAttribute(&first, CU_POINTER_ATTRIBUTE_BUFFER_ID,
                              block.base) != CUDA_SUCCESS ||
        cuPointerGetAttribute(&last, CU_POINTER_ATTRIBUTE_BUFFER_ID,
                              block.base + block.size - 1) != CUDA_SUCCESS ||
        first != last) {
        why = "the address range spans several allocations";
        return false;
    }
    memset(block.handle, 0, kIpcHandleSize);
    memcpy(block.handle, kVmmHandleTag, sizeof(kVmmHandleTag));
    memcpy(block.handle + sizeof(kVmmHandleTag), &first, sizeof(first));
    block.vmm = true;
    return true;
}

// The driver API, not cudaGetDeviceProperties: the CUDA runtime keeps device
// properties in process memory, so a process restored from a CRIU +
// cuda-checkpoint snapshot onto another GPU still gets the source GPU's UUID
// (and PCI bus id) from it, and the daemon refuses that GPU as not visible.
// cuDeviceGetUuid_v2 asks the driver, which reports the GPU the process runs
// on after the restore. The cache lives in this transport instance; a restore
// creates a new one.
bool NvlinkProxyTransport::deviceUuid(int dev, uint8_t out[kUuidSize]) {
    std::lock_guard<std::mutex> lock(uuid_mu_);
    auto it = uuids_.find(dev);
    if (it == uuids_.end()) {
        CUdevice cu_dev;
        CUuuid uuid;
        CUresult res = cuDeviceGet(&cu_dev, dev);
        if (res == CUDA_SUCCESS) res = cuDeviceGetUuid_v2(&uuid, cu_dev);
        if (res != CUDA_SUCCESS) {
            const char *name = nullptr;
            cuGetErrorName(res, &name);
            LOG(WARNING) << "nvlink_proxy: cannot read the UUID of GPU " << dev
                         << ": " << (name ? name : "unknown CUresult");
            return false;
        }
        std::array<uint8_t, kUuidSize> u;
        memcpy(u.data(), uuid.bytes, kUuidSize);
        it = uuids_.emplace(dev, u).first;
    }
    memcpy(out, it->second.data(), kUuidSize);
    return true;
}

bool NvlinkProxyTransport::describeBlock(void *addr, Block &block,
                                         std::string &why) {
    DeviceScope scope(block.device);
    if (scope.result() != CUDA_SUCCESS) {
        why = "cannot make the GPU's context current (CUresult " +
              std::to_string(static_cast<int>(scope.result())) + ")";
        return false;
    }
    CUdeviceptr base_ptr = 0;
    size_t alloc_size = 0;
    const CUresult cu_err = cuMemGetAddressRange(
        &base_ptr, &alloc_size, reinterpret_cast<CUdeviceptr>(addr));
    if (cu_err != CUDA_SUCCESS) {
        why = "cuMemGetAddressRange: CUresult " +
              std::to_string(static_cast<int>(cu_err));
        return false;
    }
    block.base = static_cast<uint64_t>(base_ptr);
    block.size = alloc_size;
    if (!deviceUuid(block.device, block.uuid)) {
        why = "cannot read the GPU's UUID";
        return false;
    }
    // Virtual memory management memory is imported by the daemon from a
    // file descriptor (REGISTER_FD); it is never asked for a legacy IPC
    // handle.
    bool is_vmm = false;
    if (vmmBlock(block, &is_vmm, why)) {
        if (vmm_enabled_) return true;
        why =
            "virtual memory management memory, which the daemon imports "
            "with MC_NVLINK_PROXY_VMM=1";
        return false;
    }
    if (is_vmm) {
        why =
            "virtual memory management memory the daemon cannot import: " + why;
        return false;
    }
    if (require_vmm_) {
        // A process that exported memory with cudaIpcGetMemHandle cannot be
        // checkpointed by cuda-checkpoint any more, even once no importer
        // is left.
        why =
            "not virtual memory management memory (cuMemCreate with "
            "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR); it is not exported "
            "with cudaIpcGetMemHandle, which would keep cuda-checkpoint from "
            "checkpointing this process";
        return false;
    }
    cudaIpcMemHandle_t handle;
    const cudaError_t err =
        cudaIpcGetMemHandle(&handle, reinterpret_cast<void *>(base_ptr));
    if (err != cudaSuccess) {
        cudaGetLastError();
        why = std::string("cannot be exported with cudaIpcGetMemHandle (") +
              cudaGetErrorString(err) + ")";
        return false;
    }
    memcpy(block.handle, &handle, kIpcHandleSize);
    return true;
}

int NvlinkProxyTransport::install(std::string &local_server_name,
                                  std::shared_ptr<TransferMetadata> meta,
                                  std::shared_ptr<Topology> topo) {
    (void)topo;
    metadata_ = meta;
    local_server_name_ = local_server_name;
    const char *sock = getenv("MC_NVLINK_PROXY_SOCKET");
    if (!sock || !*sock) {
        LOG(ERROR) << "nvlink_proxy: MC_NVLINK_PROXY_SOCKET is not set";
        return -1;
    }
    socket_path_ = sock;
    node_id_ = resolveNodeId();
    client_id_ = nonZeroRandom();
    process_token_ = acquireProcessToken();
    timeout_ms_ = std::max(100, envInt("MC_NVLINK_PROXY_TIMEOUT_MS", 30000));
    reconnect_wait_ms_ = std::min(
        timeout_ms_, std::max(0, envInt("MC_NVLINK_PROXY_RECONNECT_WAIT_MS",
                                        timeout_ms_ / 2)));
    pair_test_wait_ms_ = std::min(
        timeout_ms_,
        std::max(0, envInt("MC_NVLINK_PROXY_PAIR_TEST_WAIT_MS", 3000)));
    release_wait_ms_ = std::min(
        600000, std::max(0, envInt("MC_NVLINK_PROXY_RELEASE_WAIT_MS", 10000)));
    require_vmm_ = envInt("MC_NVLINK_PROXY_REQUIRE_VMM", 0) != 0;
    vmm_enabled_ = require_vmm_ || envInt("MC_NVLINK_PROXY_VMM", 0) != 0;
    stats_interval_s_ = envInt("MC_NVLINK_PROXY_STATS_INTERVAL", 60);
    outage_since_ns_ = nowNs();  // until the first connection succeeds

    // Compose with the base transport's local segment (e.g. "tcp" ->
    // "tcp,nvlink_proxy") instead of replacing it.
    auto old_desc = metadata_->getSegmentDescByID(LOCAL_SEGMENT_ID);
    auto desc = std::make_shared<SegmentDesc>();
    if (old_desc) *desc = *old_desc;
    desc->name = local_server_name_;
    if (desc->protocol.empty()) {
        desc->protocol = kProtocol;
    } else if (("," + desc->protocol + ",").find(",nvlink_proxy,") ==
               std::string::npos) {
        desc->protocol += ",";
        desc->protocol += kProtocol;
    }
    metadata_->addLocalSegment(LOCAL_SEGMENT_ID, local_server_name_,
                               std::move(desc));

    bool connected;
    {
        std::lock_guard<std::mutex> lock(control_mu_);
        connected = ensureControlLocked();
    }
    health_thread_ = std::thread(&NvlinkProxyTransport::healthLoop, this);
    char id[32];
    snprintf(id, sizeof(id), "%016" PRIx64, client_id_);
    LOG(INFO) << "nvlink_proxy: installed (socket=" << socket_path_
              << ", node_id=" << node_id_ << ", client_id=" << id
              << ", timeout_ms=" << timeout_ms_
              << ", reconnect_wait_ms=" << reconnect_wait_ms_
              << ", pair_test_wait_ms=" << pair_test_wait_ms_
              << ", release_wait_ms=" << release_wait_ms_
              << ", vmm=" << (vmm_enabled_ ? 1 : 0) << ", daemon "
              << (connected ? "connected" : "not reachable yet") << ")";
    return 0;
}

void NvlinkProxyTransport::setFallbackTransport(Transport *transport,
                                                const std::string &name) {
    fallback_ = transport;
    int inflight =
        std::max(1, envInt("MC_NVLINK_PROXY_FALLBACK_INFLIGHT", 512));
    if (name == "tcp") {
        // Stay below the TCP lane's per-peer queue: requests beyond it wait
        // in a short admission queue and are then rejected (queue-full).
        const int queue =
            std::max(1, envInt("MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER", 1024));
        inflight = std::min(inflight, std::max(1, queue / 2));
    }
    fallback_submitter_ =
        std::make_unique<BoundedSubmitter>(inflight, std::min(256, inflight));
    LOG(INFO) << "nvlink_proxy: base transport " << name << ", at most "
              << inflight << " fallback request(s) in flight";
}

// ------------------------------------------------------------------ daemon

int NvlinkProxyTransport::connectSocket(int timeout_ms) {
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (socket_path_.size() >= sizeof(addr.sun_path)) {
        close(fd);
        errno = ENAMETOOLONG;
        return -1;
    }
    memcpy(addr.sun_path, socket_path_.c_str(), socket_path_.size() + 1);
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    // SO_SNDTIMEO also bounds connect() on AF_UNIX sockets.
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        int saved = errno;
        close(fd);
        errno = saved;
        return -1;
    }
    return fd;
}

bool NvlinkProxyTransport::rpc(int fd, MsgType type, const void *payload,
                               size_t len, ProxyStatus &status,
                               std::string &reply, bool *sent, int pass_fd) {
    static std::atomic<uint64_t> seq{1};
    MsgHeader h{};
    h.magic = kMagic;
    h.version = kProtocolVersion;
    h.type = static_cast<uint16_t>(type);
    h.payload_len = static_cast<uint32_t>(len);
    h.seq = seq++;
    if (sent) *sent = false;
    if (pass_fd >= 0) {
        if (!sendWithFd(fd, &h, sizeof(h), pass_fd)) return false;
    } else if (!writeAll(fd, &h, sizeof(h))) {
        return false;
    }
    if (len && !writeAll(fd, payload, len)) return false;
    if (sent) *sent = true;
    MsgHeader r;
    if (!readAll(fd, &r, sizeof(r))) return false;
    if (r.magic != kMagic || r.seq != h.seq || r.type != h.type ||
        r.payload_len > kMaxPayload) {
        errno = EPROTO;
        return false;
    }
    reply.resize(r.payload_len);
    if (r.payload_len && !readAll(fd, &reply[0], r.payload_len)) return false;
    status = static_cast<ProxyStatus>(r.status);
    return true;
}

bool NvlinkProxyTransport::hello(int fd, bool control, uint64_t &epoch) {
    HelloReq req{};
    req.client_id = client_id_;
    req.pid = static_cast<uint32_t>(getpid());
    req.flags = (control ? kHelloFlagControl : 0) | kHelloFlagProcessToken;
    strncpy(req.node_id, node_id_.c_str(), kHelloTokenOffset - 1);
    memcpy(req.node_id + kHelloTokenOffset, &process_token_,
           sizeof(process_token_));
    ProxyStatus st;
    std::string reply;
    if (!rpc(fd, MsgType::kHello, &req, sizeof(req), st, reply)) return false;
    if (st != ProxyStatus::kOk || reply.size() != sizeof(HelloResp)) {
        LOG(WARNING) << "nvlink_proxy: daemon rejected HELLO: "
                     << proxyStatusName(st) << " " << reply;
        errno = EPROTO;
        return false;
    }
    HelloResp resp;
    memcpy(&resp, reply.data(), sizeof(resp));
    epoch = resp.daemon_epoch;
    return epoch != 0;
}

void NvlinkProxyTransport::closeControlLocked() {
    if (control_fd_ >= 0) {
        close(control_fd_);
        if (ever_registered_ && epoch_.load() != 0) {
            lost_epoch_ = epoch_.load();
            lost_ns_ = nowNs();
        }
    }
    control_fd_ = -1;
    markUnhealthy();
    epoch_ = 0;
}

void NvlinkProxyTransport::markUnhealthy() {
    healthy_ = false;
    int64_t expected = 0;
    outage_since_ns_.compare_exchange_strong(expected, nowNs());
}

void NvlinkProxyTransport::markHealthy(bool reconnected) {
    {
        std::lock_guard<std::mutex> lock(state_mu_);
        if (reconnected) epoch_since_ns_ = nowNs();
        outage_since_ns_ = 0;
        healthy_ = true;
    }
    state_cv_.notify_all();
}

bool NvlinkProxyTransport::waitForDaemon(int64_t limit_ns) {
    std::unique_lock<std::mutex> lock(state_mu_);
    while (!healthy_) {
        if (shut_down_) return false;
        const int64_t now = nowNs();
        const int64_t since = outage_since_ns_.load();
        // Requests stop waiting once the outage itself is older than the
        // reconnect window, so a daemon that stays away does not delay every
        // later request by the full window.
        const int64_t window_end =
            (since ? since : now) + int64_t(reconnect_wait_ms_) * 1000000LL;
        const int64_t until = std::min(limit_ns, window_end);
        if (now >= until) return false;
        state_cv_.wait_for(lock, std::chrono::nanoseconds(until - now));
    }
    return true;
}

bool NvlinkProxyTransport::ensureControlLocked() {
    if (control_fd_ >= 0) return true;
    if (shut_down_) return false;
    int fd = connectSocket(2000);
    uint64_t epoch = 0;
    if (fd >= 0 && !hello(fd, true, epoch)) {
        int saved = errno;
        close(fd);
        fd = -1;
        errno = saved;
    }
    if (fd < 0) {
        const int err = errno;
        int64_t now = nowNs();
        if (last_unreachable_log_ns_ == 0 ||
            now - last_unreachable_log_ns_ > 30LL * 1000 * 1000 * 1000) {
            last_unreachable_log_ns_ = now;
            if (err == EACCES || err == EPERM) {
                // Not a daemon that is down or restarting: this engine may
                // not use the socket at all, and never will by retrying.
                LOG(ERROR) << "nvlink_proxy: no permission to connect to the "
                              "daemon at "
                           << socket_path_ << " (" << strerror(err)
                           << "): the daemon creates its socket owner-only "
                              "(0600) by default; run this engine as the "
                              "daemon's user, or give both a shared group "
                              "and start the daemon with --socket-mode 0660. "
                              "Same-node transfers use the base transport";
            } else {
                LOG(WARNING) << "nvlink_proxy: daemon not reachable at "
                             << socket_path_ << " (" << strerror(err)
                             << "); same-node transfers use the base "
                                "transport";
            }
        }
        errno = err;
        return false;
    }
    control_fd_ = fd;
    epoch_ = epoch;
    last_answer_ns_ = nowNs();
    size_t ok = 0, failed = 0;
    {
        std::lock_guard<std::mutex> lock(blocks_mu_);
        for (auto &kv : blocks_) {
            if (registerBlockLocked(kv.second))
                ++ok;
            else
                ++failed;
            if (control_fd_ < 0) break;  // connection died meanwhile
        }
    }
    if (control_fd_ < 0) return false;
    markHealthy(true);
    counters_.reconnects++;
    char ep[32];
    snprintf(ep, sizeof(ep), "%016" PRIx64, epoch);
    if (failed) {
        LOG(ERROR) << "nvlink_proxy: connected to daemon " << socket_path_
                   << " (epoch " << ep << "), registered " << ok
                   << " block(s); " << failed
                   << " refused by the daemon: their transfers use the base "
                      "transport (register_refusals="
                   << counters_.register_refusals.load() << ")";
    } else {
        LOG(INFO) << "nvlink_proxy: connected to daemon " << socket_path_
                  << " (epoch " << ep << "), registered " << ok << " block(s)";
    }
    return true;
}

// Requires control_mu_ and blocks_mu_.
bool NvlinkProxyTransport::registerBlockLocked(Block &block) {
    block.registered_epoch = 0;
    if (control_fd_ < 0) return false;
    RegisterReq req{};
    req.base = block.base;
    req.size = block.size;
    memcpy(req.gpu_uuid, block.uuid, kUuidSize);
    memcpy(req.ipc_handle, block.handle, kIpcHandleSize);
    int fd = -1;
    if (block.vmm) {
        DeviceScope scope(block.device);
        CUresult error = scope.result();
        if (error == CUDA_SUCCESS) fd = exportVmmFd(block.base, &error);
        if (fd < 0) {
            counters_.register_refusals++;
            LOG(ERROR) << "nvlink_proxy: cannot export block "
                       << reinterpret_cast<void *>(block.base)
                       << " as a POSIX file descriptor (CUresult " << error
                       << "); transfers from/to it use the base transport";
            return false;
        }
    }
    ProxyStatus st;
    std::string reply;
    bool sent = false;
    const bool answered =
        rpc(control_fd_, block.vmm ? MsgType::kRegisterFd : MsgType::kRegister,
            &req, sizeof(req), st, reply, &sent, fd);
    const int err = errno;
    // Sent: the daemon may hold it even if its answer got lost.
    if (sent) ever_registered_ = true;
    if (fd >= 0) close(fd);  // the daemon has its own once it got it
    if (!answered) {
        LOG(WARNING) << "nvlink_proxy: lost daemon connection while "
                        "registering: "
                     << strerror(err);
        closeControlLocked();
        return false;
    }
    if (st != ProxyStatus::kOk) {
        // Not silent: the block's transfers use the base transport from now
        // on, typically because the daemon's container does not see this
        // engine's GPU. Logged at most every 10 s, every refusal counted.
        counters_.register_refusals++;
        if (st == ProxyStatus::kUnknownDevice)
            counters_.register_refusals_not_visible++;
        const int64_t now = nowNs();
        int64_t last = last_refusal_log_ns_.load();
        if (now - last >= 10LL * 1000 * 1000 * 1000 &&
            last_refusal_log_ns_.compare_exchange_strong(last, now)) {
            const uint64_t more = suppressed_refusal_logs_.exchange(0);
            LOG(ERROR)
                << "nvlink_proxy: daemon refused block "
                << reinterpret_cast<void *>(block.base) << " (" << block.size
                << " bytes): " << proxyStatusName(st) << " " << reply
                << (st == ProxyStatus::kUnknownDevice
                        ? " (the daemon must see this engine's GPU: run it "
                          "where every engine GPU is visible, e.g. "
                          "NVIDIA_VISIBLE_DEVICES=all)"
                        : "")
                << "; transfers from/to it use the base transport"
                << (more ? " [+" + std::to_string(more) + " similar]" : "");
        } else {
            suppressed_refusal_logs_++;
        }
        return false;
    }
    block.registered_epoch = epoch_.load();
    ever_registered_ = true;
    return true;
}

bool NvlinkProxyTransport::acquireCopyConnection(Connection &conn) {
    const uint64_t epoch = epoch_.load();
    if (epoch == 0 || shut_down_) return false;
    {
        std::lock_guard<std::mutex> lock(pool_mu_);
        while (!idle_.empty()) {
            conn = idle_.back();
            idle_.pop_back();
            if (conn.epoch == epoch) return true;
            close(conn.fd);  // connection to a previous daemon instance
        }
    }
    int fd = connectSocket(timeout_ms_);
    if (fd < 0) return false;
    uint64_t got = 0;
    if (!hello(fd, false, got) || got != epoch) {
        close(fd);
        return false;
    }
    conn.fd = fd;
    conn.epoch = got;
    return true;
}

void NvlinkProxyTransport::releaseCopyConnection(Connection &conn,
                                                 bool reusable) {
    if (!reusable || shut_down_ || conn.epoch != epoch_.load()) {
        close(conn.fd);
        conn.fd = -1;
        return;
    }
    std::lock_guard<std::mutex> lock(pool_mu_);
    idle_.push_back(conn);
}

NvlinkProxyTransport::CopyOutcome NvlinkProxyTransport::copy(
    const std::vector<CopyEntry> &entries, size_t begin, size_t count,
    std::string &error, const UnconfirmedCounters &parents,
    int64_t deadline_ns) {
    Connection conn;
    if (!acquireCopyConnection(conn)) {
        error = std::string("cannot reach daemon: ") + strerror(errno);
        requestRecheck();
        return CopyOutcome::kNoDaemon;
    }
    std::string payload;
    CopyReqHeader rh{};
    rh.count = static_cast<uint32_t>(count);
    // The daemon refuses to *start* after half the timeout, so a copy we
    // gave up on cannot land after the fallback transport rewrote the data;
    // it also paces its issue to end within twice the budget, so half of
    // the time left to the batch deadline ends the copy before that.
    int64_t budget_ms = timeout_ms_ / 2;
    if (deadline_ns != 0) {
        const int64_t left_ms = (deadline_ns - nowNs()) / 1000000LL;
        budget_ms = std::min<int64_t>(budget_ms, left_ms / 2);
    }
    rh.budget_ms = static_cast<uint32_t>(std::max<int64_t>(budget_ms, 1));
    payload.reserve(sizeof(rh) + count * sizeof(CopyEntry));
    payload.append(reinterpret_cast<const char *>(&rh), sizeof(rh));
    payload.append(reinterpret_cast<const char *>(entries.data() + begin),
                   count * sizeof(CopyEntry));
    ProxyStatus st;
    std::string reply;
    bool sent = false;
    if (!rpc(conn.fd, MsgType::kCopy, payload.data(), payload.size(), st, reply,
             &sent)) {
        error = std::string("daemon connection error: ") +
                (errno == EAGAIN || errno == EWOULDBLOCK ? "timed out"
                                                         : strerror(errno));
        if (sent && !parents.empty()) {
            // The daemon got the copy but did not answer: it may still be
            // running it. Its batches stay unquiesced until the late reply
            // arrives on this connection, or a new daemon epoch shows the
            // old daemon gone.
            addUnconfirmed(Unconfirmed{conn.fd, conn.epoch, nullptr, parents});
            conn.fd = -1;
        } else {
            releaseCopyConnection(conn, false);
        }
        requestRecheck();
        return CopyOutcome::kNoDaemon;
    }
    releaseCopyConnection(conn, true);
    if (st != ProxyStatus::kOk) {
        error = proxyStatusName(st);
        if (reply.size() > sizeof(CopyResp)) {
            error += ": " + reply.substr(sizeof(CopyResp));
        }
    }
    return copyOutcome(st, nowNs() - epoch_since_ns_.load() <
                               int64_t(reconnect_wait_ms_) * 1000000LL);
}

NvlinkProxyTransport::CopyOutcome NvlinkProxyTransport::copyOutcome(
    ProxyStatus status, bool fresh) {
    switch (status) {
        case ProxyStatus::kOk:
            return CopyOutcome::kOk;
        case ProxyStatus::kUnknownSource:
        case ProxyStatus::kUnknownDestination:
            // Right after a daemon (re)start the peer may not have
            // re-registered its blocks yet: retry shortly instead of falling
            // back.
            return fresh ? CopyOutcome::kRetryLater : CopyOutcome::kFailed;
        case ProxyStatus::kPairUnderTest:
            // Refused before anything was issued, until the self-test of the
            // GPU pair decided (about a second).
            return CopyOutcome::kPairUnderTest;
        default:
            return CopyOutcome::kFailed;
    }
}

int64_t NvlinkProxyTransport::pairTestWaitEnd(int64_t start_ns,
                                              int64_t deadline_ns,
                                              int pair_test_wait_ms) {
    return std::min<int64_t>(start_ns + int64_t(pair_test_wait_ms) * 1000000LL,
                             start_ns + (deadline_ns - start_ns) / 2);
}

void NvlinkProxyTransport::requestRecheck() {
    // Hold proxy traffic until the health thread has re-validated the
    // daemon (it pings and, if needed, reconnects and re-registers).
    markUnhealthy();
    {
        std::lock_guard<std::mutex> lock(health_mu_);
        recheck_ = true;
    }
    health_cv_.notify_all();
}

void NvlinkProxyTransport::healthLoop() {
    int backoff_ms = 100;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(health_mu_);
            int wait_ms = healthy_ ? 1000 : backoff_ms;
            health_cv_.wait_for(lock, std::chrono::milliseconds(wait_ms),
                                [this] { return stop_ || recheck_; });
            if (stop_) break;
            recheck_ = false;
        }
        bool ok;
        {
            std::lock_guard<std::mutex> lock(control_mu_);
            ok = control_fd_ >= 0;
            if (ok) {
                ProxyStatus st;
                std::string reply;
                ok = rpc(control_fd_, MsgType::kPing, nullptr, 0, st, reply) &&
                     st == ProxyStatus::kOk;
                if (ok) last_answer_ns_ = nowNs();
                if (!ok) {
                    LOG(WARNING)
                        << "nvlink_proxy: lost daemon connection ("
                        << strerror(errno)
                        << "); same-node requests wait up to "
                        << reconnect_wait_ms_
                        << " ms for it before using the base transport";
                    closeControlLocked();
                } else if (!healthy_) {
                    markHealthy(false);
                }
            }
            if (!ok) ok = ensureControlLocked();
        }
        // Requests may be waiting for the daemon: retry the connection
        // often (a failed connect on a unix socket is cheap).
        backoff_ms = ok ? 100 : std::min(backoff_ms * 2, 400);
        reapUnconfirmed();
        maybeLogStats(false);
    }
}

void NvlinkProxyTransport::maybeLogStats(bool force) {
    const int64_t now = nowNs();
    std::lock_guard<std::mutex> lock(stats_mu_);
    if (!force) {
        if (stats_interval_s_ <= 0 || now < next_stats_ns_) return;
    }
    next_stats_ns_ =
        now + int64_t(std::max(stats_interval_s_, 1)) * 1000000000LL;
    size_t blocks = 0, registered = 0;
    {
        std::lock_guard<std::mutex> bl(blocks_mu_);
        const uint64_t epoch = epoch_.load();
        blocks = blocks_.size();
        for (auto &kv : blocks_)
            if (epoch && kv.second.registered_epoch == epoch) ++registered;
    }
    const uint64_t batches = counters_.proxy_batches.load();
    char line[1280];
    snprintf(
        line, sizeof(line),
        "healthy=%d proxied_requests=%" PRIu64 " proxied_bytes=%" PRIu64
        " proxied_batches=%" PRIu64 " avg_batch_us=%" PRIu64
        " max_batch_us=%" PRIu64 " held_requests=%" PRIu64
        " max_hold_ms=%" PRIu64 " retried_requests=%" PRIu64
        " fallback_requests=%" PRIu64 " (daemon_unavailable=%" PRIu64
        " daemon_error=%" PRIu64 " not_servable=%" PRIu64 " pair_test=%" PRIu64
        ") fallback_retries=%" PRIu64 " failed_requests=%" PRIu64
        " remote_node_requests=%" PRIu64
        " registered_blocks=%zu/%zu register_refusals=%" PRIu64
        " (not_visible=%" PRIu64 ") connects=%" PRIu64,
        healthy_.load() ? 1 : 0, counters_.proxy_requests.load(),
        counters_.proxy_bytes.load(), batches,
        batches ? counters_.proxy_us_total.load() / batches : 0,
        counters_.proxy_us_max.load(), counters_.held_requests.load(),
        counters_.held_us_max.load() / 1000, counters_.retried_requests.load(),
        counters_.fallback_requests.load(), counters_.fallback_unhealthy.load(),
        counters_.fallback_daemon_error.load(),
        counters_.fallback_untranslatable.load(),
        counters_.fallback_pair_test.load(), counters_.fallback_retries.load(),
        counters_.failed_requests.load(), counters_.remote_node_requests.load(),
        registered, blocks, counters_.register_refusals.load(),
        counters_.register_refusals_not_visible.load(),
        counters_.reconnects.load());
    if (!force && last_stats_ == line) return;  // nothing changed
    last_stats_ = line;
    LOG(INFO) << "nvlink_proxy stats: " << line;
}

// ------------------------------------------------------------- translation

NvlinkProxyTransport::LocalBlockState NvlinkProxyTransport::findLocalBlock(
    uint64_t addr, uint64_t length, uint64_t &base) {
    std::lock_guard<std::mutex> lock(blocks_mu_);
    auto it = blocks_.upper_bound(addr);
    if (it == blocks_.begin()) return LocalBlockState::kNone;
    --it;
    const Block &b = it->second;
    if (addr < b.base || addr - b.base > b.size ||
        length > b.size - (addr - b.base))
        return LocalBlockState::kNone;
    base = b.base;
    const uint64_t epoch = epoch_.load();
    return (epoch != 0 && b.registered_epoch == epoch)
               ? LocalBlockState::kRegistered
               : LocalBlockState::kUnregistered;
}

bool NvlinkProxyTransport::canRoute(const TransferRequest &request,
                                    const BufferDesc &target_buffer) {
    const char *node;
    size_t node_len;
    uint64_t client, base, size;
    if (!decodeBufferRef(target_buffer.shm_name, node, node_len, client, base,
                         size))
        return false;
    if (node_len != node_id_.size() ||
        memcmp(node, node_id_.data(), node_len) != 0) {
        counters_.remote_node_requests++;
        return false;
    }
    uint64_t local_base;
    const uint64_t local = reinterpret_cast<uint64_t>(request.source);
    // A local address outside every registered device block (e.g. host
    // memory) is simply not proxy traffic.
    const LocalBlockState state =
        findLocalBlock(local, request.length, local_base);
    if (state == LocalBlockState::kNone) return false;
    if (healthy_ && state == LocalBlockState::kRegistered) return true;
    // While the daemon is unavailable the proxy transport keeps same-node
    // GPU traffic: it waits for the daemon to come back and otherwise moves
    // the requests onto the base transport with backpressure, instead of
    // handing whole batches to the base transport at once.
    if (!healthy_) return true;
    // The daemon is up but refused this block (e.g. GPU compute mode).
    counters_.fallback_requests++;
    counters_.fallback_untranslatable++;
    return false;
}

const NvlinkProxyTransport::TargetIndex *NvlinkProxyTransport::targetIndex(
    std::vector<TargetIndex> &cache, SegmentID target_id) {
    for (auto &t : cache)
        if (t.id == target_id) return &t;
    auto desc = metadata_->getSegmentDescByID(target_id);
    if (!desc) return nullptr;
    TargetIndex t;
    t.id = target_id;
    for (const auto &buffer : desc->buffers) {
        if (buffer.protocol != kProtocol) continue;
        TargetIndex::Ref r;
        const char *node;
        size_t node_len;
        if (!decodeBufferRef(buffer.shm_name, node, node_len, r.client, r.base,
                             r.size))
            continue;
        if (node_len != node_id_.size() ||
            memcmp(node, node_id_.data(), node_len) != 0)
            continue;
        r.addr = buffer.addr;
        r.length = buffer.length;
        t.refs.push_back(r);
    }
    std::sort(t.refs.begin(), t.refs.end(),
              [](const TargetIndex::Ref &a, const TargetIndex::Ref &b) {
                  return a.addr < b.addr;
              });
    cache.push_back(std::move(t));
    return &cache.back();
}

NvlinkProxyTransport::TranslateResult NvlinkProxyTransport::translate(
    const TransferRequest &request, const TargetIndex &target,
    CopyEntry &entry) {
    using R = TranslateResult;
    const uint64_t remote = request.target_offset;
    // Last published buffer starting at or below the target address.
    auto it = std::upper_bound(
        target.refs.begin(), target.refs.end(), remote,
        [](uint64_t a, const TargetIndex::Ref &r) { return a < r.addr; });
    if (it == target.refs.begin()) return R::kNotServable;
    const TargetIndex::Ref &ref = *(it - 1);
    if (remote - ref.addr > ref.length ||
        request.length > ref.length - (remote - ref.addr))
        return R::kNotServable;
    if (remote < ref.base || remote - ref.base > ref.size ||
        request.length > ref.size - (remote - ref.base))
        return R::kNotServable;
    uint64_t local_base;
    const uint64_t local = reinterpret_cast<uint64_t>(request.source);
    switch (findLocalBlock(local, request.length, local_base)) {
        case LocalBlockState::kRegistered:
            break;
        case LocalBlockState::kNone:
            return R::kNotServable;
        case LocalBlockState::kUnregistered: {
            // Being re-registered with a daemon that just (re)started; a
            // block the daemon refused under an established epoch is not
            // servable.
            const int64_t since = epoch_since_ns_.load();
            if (!healthy_ || epoch_.load() == 0 ||
                nowNs() - since < int64_t(reconnect_wait_ms_) * 1000000LL)
                return R::kRetry;
            return R::kNotServable;
        }
    }
    entry.length = request.length;
    if (request.opcode == TransferRequest::WRITE) {
        entry.src_client = client_id_;
        entry.src_base = local_base;
        entry.src_offset = local - local_base;
        entry.dst_client = ref.client;
        entry.dst_base = ref.base;
        entry.dst_offset = remote - ref.base;
    } else {
        entry.src_client = ref.client;
        entry.src_base = ref.base;
        entry.src_offset = remote - ref.base;
        entry.dst_client = client_id_;
        entry.dst_base = local_base;
        entry.dst_offset = local - local_base;
    }
    return R::kOk;
}

// ------------------------------------------------------------- submission

void NvlinkProxyTransport::finishTask(TransferTask *task, bool ok) {
    const auto &request = *task->request;
    task->total_bytes = request.length;
    Slice *slice = getSliceCache().allocate();
    slice->source_addr = request.source;
    slice->length = request.length;
    slice->opcode = request.opcode;
    slice->target_id = request.target_id;
    slice->task = task;
    slice->status = Slice::PENDING;
    slice->ts = 0;
    task->slice_list.push_back(slice);
    __sync_fetch_and_add(&task->slice_count, 1);
    if (ok)
        slice->markSuccess();
    else
        slice->markFailed();
}

void NvlinkProxyTransport::completeTask(TransferTask *task) {
    finishTask(task, true);
}

void NvlinkProxyTransport::failTask(TransferTask *task) {
    finishTask(task, false);
}

void NvlinkProxyTransport::fallback(const std::vector<TransferTask *> &tasks,
                                    FallbackReason reason,
                                    const std::string &detail,
                                    int64_t deadline_ns) {
    if (tasks.empty()) return;
    counters_.fallback_requests += tasks.size();
    switch (reason) {
        case FallbackReason::kUnhealthy:
            counters_.fallback_unhealthy += tasks.size();
            break;
        case FallbackReason::kDaemonError:
            counters_.fallback_daemon_error += tasks.size();
            break;
        case FallbackReason::kUntranslatable:
            counters_.fallback_untranslatable += tasks.size();
            break;
        case FallbackReason::kPairTest:
            counters_.fallback_pair_test += tasks.size();
            break;
    }
    int64_t now = nowNs();
    int64_t last = last_fallback_log_ns_.load();
    if (now - last >= kFallbackLogIntervalNs &&
        last_fallback_log_ns_.compare_exchange_strong(last, now)) {
        LOG(WARNING) << "nvlink_proxy: " << tasks.size()
                     << " request(s) fall back to the base transport ("
                     << reasonName(static_cast<int>(reason))
                     << (detail.empty() ? "" : ": " + detail)
                     << "); fallback_requests="
                     << counters_.fallback_requests.load();
    }
    if (!fallback_ || !fallback_submitter_) {
        for (auto *task : tasks) failTask(task);
        counters_.failed_requests += tasks.size();
        return;
    }
    std::vector<const TransferRequest *> requests(tasks.size());
    for (size_t i = 0; i < tasks.size(); ++i) requests[i] = tasks[i]->request;
    // The caller's batch deadline bounds the fallback too.
    for (auto *task : tasks) {
        if (!task->batch_id) continue;
        const int64_t d =
            Transport::toBatchDesc(task->batch_id).control.deadlineNs();
        if (d != 0 && d < deadline_ns) deadline_ns = d;
    }
    const int64_t abandon_ns =
        deadline_ns + std::max<int64_t>(timeout_ms_, 10000) * 1000000LL;
    const UnconfirmedCounters parents = countersOf(tasks, 0, tasks.size());
    const UnconfirmedCounters parents_unprovable =
        countersOf(tasks, 0, tasks.size(), true);
    auto res = fallback_submitter_->run(
        fallback_, requests, deadline_ns, abandon_ns,
        [&](size_t i, bool ok) {
            ok ? completeTask(tasks[i]) : failTask(tasks[i]);
        },
        [this, &parents, &parents_unprovable](Transport::BatchDesc *batch) {
            // Not settled yet: the caller's batches stay unquiesced until
            // it is (the health thread frees it then).
            Unconfirmed u{-1, 0, batch, parents};
            u.parents_unprovable = parents_unprovable;
            addUnconfirmed(std::move(u));
        },
        [&tasks]() {
            for (auto *task : tasks)
                if (task->batch_id &&
                    Transport::toBatchDesc(task->batch_id).control.canceled())
                    return true;
            return false;
        },
        [&parents, &parents_unprovable]() {
            // An attempt settled with a write its receiver can never confirm
            // (TCP receiver older than protocol v3): so are the caller's
            // batches (counted unconfirmed first, then unprovable).
            for (auto &counter : parents)
                counter->fetch_add(1, std::memory_order_acq_rel);
            for (auto &counter : parents_unprovable)
                counter->fetch_add(1, std::memory_order_acq_rel);
        });
    counters_.fallback_retries += res.retried;
    if (res.failed) {
        counters_.failed_requests += res.failed;
        LOG(ERROR) << "nvlink_proxy: " << res.failed << " of " << tasks.size()
                   << " request(s) failed on the base transport within the "
                      "request budget ("
                   << timeout_ms_ << " ms)";
    }
}

Status NvlinkProxyTransport::submitTransferTask(
    const std::vector<TransferTask *> &task_list) {
    const int64_t start = nowNs();
    const int64_t hold_deadline =
        start + int64_t(reconnect_wait_ms_) * 1000000LL;
    const int64_t deadline = start + int64_t(timeout_ms_) * 1000000LL;
    // The latest a request may take: its budget, or its batch's deadline
    // when that is earlier.
    auto requestDeadline = [deadline](TransferTask *task) {
        int64_t d = deadline;
        if (task->batch_id) {
            const int64_t b =
                Transport::toBatchDesc(task->batch_id).control.deadlineNs();
            if (b != 0 && b < d) d = b;
        }
        return d;
    };

    std::vector<TransferTask *> pending(task_list.begin(), task_list.end());
    std::vector<TransferTask *> not_servable, failed, unavailable, pair_late;
    std::string error, pair_error;
    bool held = false;
    int round = 0;
    while (!pending.empty()) {
        // The caller's batch was canceled or ran out of time: nothing more
        // of it is sent (neither to the daemon nor to the base transport).
        {
            const int64_t now_ns = nowNs();
            std::vector<TransferTask *> live;
            live.reserve(pending.size());
            for (auto *task : pending) {
                if (task->batch_id && Transport::toBatchDesc(task->batch_id)
                                          .control.expired(now_ns)) {
                    failTask(task);
                    counters_.failed_requests++;
                } else {
                    live.push_back(task);
                }
            }
            pending.swap(live);
            if (pending.empty()) break;
        }
        if (!healthy_) {
            if (!held) {
                held = true;
                counters_.held_requests += pending.size();
            }
            if (!waitForDaemon(hold_deadline)) {
                unavailable.swap(pending);
                break;
            }
            continue;
        }

        std::vector<CopyEntry> entries;
        std::vector<TransferTask *> proxied, retry, pair_wait;
        bool retry_later = false;
        entries.reserve(pending.size());
        proxied.reserve(pending.size());
        std::vector<TargetIndex> targets;
        const TargetIndex *last_target = nullptr;
        for (auto *task : pending) {
            CopyEntry e{};
            const TargetIndex *target = nullptr;
            if (task->request) {
                if (last_target && last_target->id == task->request->target_id)
                    target = last_target;
                else
                    target = last_target =
                        targetIndex(targets, task->request->target_id);
            }
            const TranslateResult tr =
                target ? translate(*task->request, *target, e)
                       : TranslateResult::kNotServable;
            if (tr == TranslateResult::kNotServable) {
                not_servable.push_back(task);
                continue;
            }
            if (tr == TranslateResult::kRetry) {
                retry.push_back(task);
                retry_later = true;
                continue;
            }
            entries.push_back(e);
            proxied.push_back(task);
        }

        for (size_t begin = 0; begin < proxied.size();
             begin += kMaxEntriesPerCopy) {
            const size_t count =
                std::min(kMaxEntriesPerCopy, proxied.size() - begin);
            const int64_t t0 = nowNs();
            std::string err;
            int64_t copy_deadline = 0;
            for (size_t i = begin; i < begin + count; ++i) {
                if (!proxied[i]->batch_id) continue;
                const int64_t d = Transport::toBatchDesc(proxied[i]->batch_id)
                                      .control.deadlineNs();
                if (d != 0 && (copy_deadline == 0 || d < copy_deadline))
                    copy_deadline = d;
            }
            const CopyOutcome outcome =
                copy(entries, begin, count, err,
                     countersOf(proxied, begin, count), copy_deadline);
            if (outcome != CopyOutcome::kOk) {
                const bool pair = outcome == CopyOutcome::kPairUnderTest;
                auto &dst = outcome == CopyOutcome::kFailed ? failed
                            : pair                          ? pair_wait
                                                            : retry;
                dst.insert(dst.end(), proxied.begin() + begin,
                           proxied.begin() + begin + count);
                retry_later |= outcome == CopyOutcome::kRetryLater;
                (pair ? pair_error : error) = err;
                continue;
            }
            const uint64_t us = static_cast<uint64_t>((nowNs() - t0) / 1000);
            uint64_t bytes = 0;
            for (size_t i = begin; i < begin + count; ++i) {
                bytes += entries[i].length;
                completeTask(proxied[i]);
            }
            counters_.proxy_requests += count;
            counters_.proxy_bytes += bytes;
            counters_.proxy_batches++;
            counters_.proxy_us_total += us;
            uint64_t cur = counters_.proxy_us_max.load();
            while (us > cur &&
                   !counters_.proxy_us_max.compare_exchange_weak(cur, us)) {
            }
        }
        // Daemon restarted or a peer not re-registered yet: retry through
        // the proxy while the reconnect window lasts.
        const int64_t now = nowNs();
        if (!retry.empty() && now >= hold_deadline) {
            failed.insert(failed.end(), retry.begin(), retry.end());
            retry.clear();
        }
        int64_t wake = retry.empty() ? INT64_MAX : hold_deadline;
        // The GPU pair is under the daemon's self-test (nothing of the copy
        // was issued): send it again until the test decided, while the
        // request's pair-test wait lasts; then the base transport.
        size_t pair_waiting = 0;
        for (auto *task : pair_wait) {
            const int64_t end = pairTestWaitEnd(start, requestDeadline(task),
                                                pair_test_wait_ms_);
            if (now >= end) {
                pair_late.push_back(task);
                continue;
            }
            retry.push_back(task);
            ++pair_waiting;
            wake = std::min(wake, end);
        }
        if (pair_waiting && !held) {
            held = true;
            counters_.held_requests += pair_waiting;
        }
        pending.swap(retry);
        if (pending.empty()) break;
        counters_.retried_requests += pending.size();
        if (retry_later || pair_waiting) {
            // A self-test decides within about a second: look again soon.
            const int64_t cap_ms = pair_waiting ? 50 : 200;
            const int64_t backoff_ns =
                std::min<int64_t>(cap_ms, 10LL << std::min(round, 5)) *
                1000000LL;
            // Wake up for the earliest batch deadline too: its requests
            // fail there, nothing of them is sent any more.
            for (auto *task : pending)
                wake = std::min(wake, requestDeadline(task));
            std::this_thread::sleep_for(std::chrono::nanoseconds(
                std::max<int64_t>(0, std::min(backoff_ns, wake - now))));
        }
        ++round;
    }
    if (held) {
        const uint64_t waited = static_cast<uint64_t>((nowNs() - start) / 1000);
        uint64_t cur = counters_.held_us_max.load();
        while (waited > cur &&
               !counters_.held_us_max.compare_exchange_weak(cur, waited)) {
        }
    }

    fallback(unavailable, FallbackReason::kUnhealthy,
             "daemon unavailable for longer than the reconnect window",
             deadline);
    fallback(not_servable, FallbackReason::kUntranslatable,
             "target or source not registered with the proxy", deadline);
    fallback(pair_late, FallbackReason::kPairTest,
             pair_error + "; not decided within the pair-test wait", deadline);
    fallback(failed, FallbackReason::kDaemonError, error, deadline);
    return Status::OK();
}

NvlinkProxyTransport::UnconfirmedCounters NvlinkProxyTransport::countersOf(
    const std::vector<TransferTask *> &tasks, size_t begin, size_t count,
    bool unprovable) {
    UnconfirmedCounters out;
    for (size_t i = begin; i < begin + count && i < tasks.size(); ++i) {
        if (!tasks[i] || !tasks[i]->batch_id) continue;
        const auto &control =
            Transport::toBatchDesc(tasks[i]->batch_id).control;
        auto counter =
            unprovable ? control.unprovable() : control.unconfirmed();
        if (std::find(out.begin(), out.end(), counter) == out.end())
            out.push_back(counter);
    }
    return out;
}

void NvlinkProxyTransport::addUnconfirmed(Unconfirmed u) {
    // Counted before the tasks turn terminal (the caller completes them
    // after this returns), so a batch never looks quiesced in between.
    for (auto &counter : u.parents)
        counter->fetch_add(1, std::memory_order_acq_rel);
    std::lock_guard<std::mutex> lock(unconfirmed_mu_);
    unconfirmed_.push_back(std::move(u));
}

void NvlinkProxyTransport::reapUnconfirmed() {
    std::lock_guard<std::mutex> lock(unconfirmed_mu_);
    const uint64_t epoch = epoch_.load();
    for (auto it = unconfirmed_.begin(); it != unconfirmed_.end();) {
        bool confirmed = false;
        bool unprovable = false;
        if (it->batch) {
            confirmed = BoundedSubmitter::settled(*it->batch);
            if (confirmed) {
                // Never to be confirmed: the caller's batches stay counted
                // in unconfirmed, and now also in unprovable.
                unprovable = it->batch->control.unprovable()->load(
                                 std::memory_order_acquire) > 0;
                delete it->batch;
                it->batch = nullptr;
            }
        } else {
            if (it->fd >= 0) {
                pollfd p{it->fd, POLLIN, 0};
                if (poll(&p, 1, 0) > 0) {
                    char c;
                    const ssize_t n =
                        recv(it->fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
                    if (n > 0) {
                        confirmed = true;  // the late reply: copy finished
                    } else if (n == 0 ||
                               (errno != EAGAIN && errno != EWOULDBLOCK &&
                                errno != EINTR)) {
                        // Closed without a reply: the copy returned, or the
                        // daemon is exiting (a draining daemon closes it
                        // only when it exits).
                        close(it->fd);
                        it->fd = -1;
                        it->closed_ns = nowNs();
                    }
                }
            }
            // A new daemon epoch since: the old daemon is gone. The same
            // daemon answering since: it is alive, so the close came from
            // the connection's thread after the copy returned.
            if (!confirmed && it->closed_ns != 0 && epoch != 0 &&
                (epoch != it->epoch || last_answer_ns_.load() > it->closed_ns))
                confirmed = true;
        }
        if (!confirmed) {
            ++it;
            continue;
        }
        if (it->fd >= 0) close(it->fd);
        if (unprovable) {
            for (auto &counter : it->parents_unprovable)
                counter->fetch_add(1, std::memory_order_acq_rel);
        } else {
            for (auto &counter : it->parents)
                counter->fetch_sub(1, std::memory_order_acq_rel);
        }
        it = unconfirmed_.erase(it);
    }
}

Status NvlinkProxyTransport::submitTransfer(
    BatchID batch_id, const std::vector<TransferRequest> &entries) {
    auto &batch_desc = toBatchDesc(batch_id);
    if (batch_desc.task_list.size() + entries.size() > batch_desc.batch_size) {
        return Status::InvalidArgument(
            "nvlink_proxy: exceed the limitation of capacity, batch id: " +
            std::to_string(batch_id));
    }
    size_t task_id = batch_desc.task_list.size();
    batch_desc.task_list.resize(task_id + entries.size());
    std::vector<TransferTask *> tasks;
    tasks.reserve(entries.size());
    for (const auto &request : entries) {
        auto &task = batch_desc.task_list[task_id++];
        task.batch_id = batch_id;
        task.request = &request;
        task.transport_ = this;
        tasks.push_back(&task);
    }
    return submitTransferTask(tasks);
}

Status NvlinkProxyTransport::getTransferStatus(BatchID batch_id, size_t task_id,
                                               TransferStatus &status) {
    auto &batch_desc = toBatchDesc(batch_id);
    if (task_id >= batch_desc.task_list.size()) {
        return Status::InvalidArgument(
            "nvlink_proxy: invalid task id, batch id: " +
            std::to_string(batch_id));
    }
    auto &task = batch_desc.task_list[task_id];
    if (task.transport_ && task.transport_ != this) {
        return task.transport_->getTransferStatus(batch_id, task_id, status);
    }
    status.transferred_bytes = task.transferred_bytes;
    const uint64_t success = task.success_slice_count;
    const uint64_t failed = task.failed_slice_count;
    if (success + failed == task.slice_count) {
        status.s =
            failed ? TransferStatusEnum::FAILED : TransferStatusEnum::COMPLETED;
        task.is_finished = true;
    } else {
        status.s = TransferStatusEnum::WAITING;
    }
    return Status::OK();
}

// ----------------------------------------------------------- registration

int NvlinkProxyTransport::registerLocalMemory(void *addr, size_t length,
                                              const std::string &location,
                                              bool remote_accessible,
                                              bool update_metadata) {
    (void)remote_accessible;
    // Host memory stays on the base transport only; that is not an error.
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, addr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    if (attr.type != cudaMemoryTypeDevice) return 0;
    Block fresh;
    fresh.device = attr.device;
    std::string why;
    if (!describeBlock(addr, fresh, why)) {
        // Device memory the daemon cannot import stays on the base transport
        // only; with MC_NVLINK_PROXY_REQUIRE_VMM that is an error.
        if (require_vmm_) {
            LOG(ERROR) << "nvlink_proxy: refusing to register buffer " << addr
                       << " of GPU " << attr.device << ": " << why
                       << " (MC_NVLINK_PROXY_REQUIRE_VMM=1)";
            return ERR_INVALID_ARGUMENT;
        }
        LOG(WARNING) << "nvlink_proxy: buffer " << addr << " of GPU "
                     << attr.device << ": " << why
                     << "; it stays on the base transport";
        return 0;
    }

    BufferDesc desc;
    desc.name = location.empty() ? "*" : location;
    desc.addr = reinterpret_cast<uint64_t>(addr);
    desc.length = length;
    desc.protocol = kProtocol;
    BufferRef ref;
    ref.node_id = node_id_;
    ref.client_id = client_id_;
    ref.base = fresh.base;
    ref.size = fresh.size;
    desc.shm_name = encodeBufferRef(ref);

    std::lock_guard<std::mutex> control_lock(control_mu_);
    // Publish outside blocks_mu_ so a slow metadata store cannot stall
    // request routing. A peer that sees the buffer before the daemon knows
    // the block just falls back once.
    int rc = metadata_->addLocalMemoryBuffer(desc, update_metadata);
    if (rc) return rc;
    {
        std::lock_guard<std::mutex> lock(blocks_mu_);
        bool need_register = false;
        auto it = blocks_.find(fresh.base);
        if (it == blocks_.end()) {
            it = blocks_.emplace(fresh.base, fresh).first;
            need_register = true;
        } else if (it->second.size != fresh.size ||
                   memcmp(it->second.handle, fresh.handle, kIpcHandleSize) ||
                   memcmp(it->second.uuid, fresh.uuid, kUuidSize)) {
            // The block was freed and reallocated at the same address.
            size_t users = it->second.users;
            it->second = fresh;
            it->second.users = users;
            need_register = true;
        }
        it->second.users++;
        buffers_[desc.addr] = fresh.base;
        if (need_register && control_fd_ >= 0) {
            registerBlockLocked(it->second);
        }
    }
    if (control_fd_ < 0) ensureControlLocked();  // registers all blocks
    return 0;
}

int NvlinkProxyTransport::unregisterLocalMemory(void *addr,
                                                bool update_metadata) {
    std::lock_guard<std::mutex> control_lock(control_mu_);
    uint64_t base;
    {
        std::lock_guard<std::mutex> lock(blocks_mu_);
        auto bit = buffers_.find(reinterpret_cast<uint64_t>(addr));
        if (bit == buffers_.end()) return 0;  // never published by us
        base = bit->second;
        buffers_.erase(bit);
    }
    int rc =
        metadata_->removeLocalMemoryBuffer(addr, update_metadata, kProtocol);
    std::lock_guard<std::mutex> lock(blocks_mu_);
    auto it = blocks_.find(base);
    if (it != blocks_.end() && --it->second.users == 0) {
        if (control_fd_ >= 0 && it->second.registered_epoch == epoch_.load()) {
            UnregisterReq req{base};
            ProxyStatus st;
            std::string reply;
            if (!rpc(control_fd_, MsgType::kUnregister, &req, sizeof(req), st,
                     reply)) {
                closeControlLocked();
            }
        }
        blocks_.erase(it);
    }
    return rc == ERR_ADDRESS_NOT_REGISTERED ? 0 : rc;
}

int NvlinkProxyTransport::registerLocalMemoryBatch(
    const std::vector<BufferEntry> &buffer_list, const std::string &location) {
    for (auto &buffer : buffer_list) {
        int ret = registerLocalMemory(buffer.addr, buffer.length, location,
                                      true, false);
        if (ret) return ret;
    }
    return metadata_->updateLocalSegmentDesc();
}

int NvlinkProxyTransport::unregisterLocalMemoryBatch(
    const std::vector<void *> &addr_list) {
    int first_error = 0;
    for (auto &addr : addr_list) {
        int ret = unregisterLocalMemory(addr, false);
        if (ret && !first_error) first_error = ret;
    }
    int metadata_ret = metadata_->updateLocalSegmentDesc();
    return first_error ? first_error : metadata_ret;
}

}  // namespace mooncake
