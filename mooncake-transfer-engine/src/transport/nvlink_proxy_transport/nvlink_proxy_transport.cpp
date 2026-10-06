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
#include <glog/logging.h>
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
#include <random>
#include <thread>

#include "common.h"
#include "error.h"

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

uint64_t randomClientId() {
    std::random_device rd;
    uint64_t id = (uint64_t(rd()) << 32) ^ rd() ^ (uint64_t(getpid()) << 16) ^
                  uint64_t(nowNs());
    return id ? id : 1;
}

bool deviceUuid(int dev, uint8_t out[kUuidSize]) {
    static std::mutex mu;
    static std::unordered_map<int, std::array<uint8_t, kUuidSize>> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(dev);
    if (it == cache.end()) {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            cudaGetLastError();
            LOG(WARNING) << "nvlink_proxy: cudaGetDeviceProperties(" << dev
                         << "): " << cudaGetErrorString(err);
            return false;
        }
        std::array<uint8_t, kUuidSize> u;
        memcpy(u.data(), prop.uuid.bytes, kUuidSize);
        it = cache.emplace(dev, u).first;
    }
    memcpy(out, it->second.data(), kUuidSize);
    return true;
}

const char *reasonName(int reason) {
    switch (reason) {
        case 0:
            return "daemon unavailable";
        case 1:
            return "daemon error";
        default:
            return "request not servable by the proxy";
    }
}

}  // namespace

NvlinkProxyTransport::NvlinkProxyTransport() = default;

NvlinkProxyTransport::~NvlinkProxyTransport() {
    {
        std::lock_guard<std::mutex> lock(health_mu_);
        stop_ = true;
    }
    health_cv_.notify_all();
    if (health_thread_.joinable()) health_thread_.join();
    maybeLogStats(true);
    {
        std::lock_guard<std::mutex> lock(control_mu_);
        closeControlLocked();
    }
    std::lock_guard<std::mutex> lock(pool_mu_);
    for (auto &c : idle_) close(c.fd);
    idle_.clear();
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
    client_id_ = randomClientId();
    timeout_ms_ = std::max(100, envInt("MC_NVLINK_PROXY_TIMEOUT_MS", 30000));
    reconnect_wait_ms_ = std::min(
        timeout_ms_, std::max(0, envInt("MC_NVLINK_PROXY_RECONNECT_WAIT_MS",
                                        timeout_ms_ / 2)));
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
              << ", reconnect_wait_ms=" << reconnect_wait_ms_ << ", daemon "
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
                               std::string &reply) {
    static std::atomic<uint64_t> seq{1};
    MsgHeader h{};
    h.magic = kMagic;
    h.version = kProtocolVersion;
    h.type = static_cast<uint16_t>(type);
    h.payload_len = static_cast<uint32_t>(len);
    h.seq = seq++;
    if (!writeAll(fd, &h, sizeof(h))) return false;
    if (len && !writeAll(fd, payload, len)) return false;
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
    req.flags = control ? kHelloFlagControl : 0;
    strncpy(req.node_id, node_id_.c_str(), kNodeIdSize - 1);
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
    if (control_fd_ >= 0) close(control_fd_);
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
    int fd = connectSocket(2000);
    uint64_t epoch = 0;
    if (fd >= 0 && !hello(fd, true, epoch)) {
        int saved = errno;
        close(fd);
        fd = -1;
        errno = saved;
    }
    if (fd < 0) {
        static std::atomic<int64_t> last_log{0};
        int64_t now = nowNs();
        if (now - last_log.load() > 30LL * 1000 * 1000 * 1000) {
            last_log = now;
            LOG(WARNING) << "nvlink_proxy: daemon not reachable at "
                         << socket_path_ << " (" << strerror(errno)
                         << "); same-node transfers use the base transport";
        }
        return false;
    }
    control_fd_ = fd;
    epoch_ = epoch;
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
    LOG(INFO) << "nvlink_proxy: connected to daemon " << socket_path_
              << " (epoch " << ep << "), registered " << ok << " block(s)"
              << (failed ? ", " + std::to_string(failed) + " failed" : "");
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
    ProxyStatus st;
    std::string reply;
    if (!rpc(control_fd_, MsgType::kRegister, &req, sizeof(req), st, reply)) {
        LOG(WARNING) << "nvlink_proxy: lost daemon connection while "
                        "registering: "
                     << strerror(errno);
        closeControlLocked();
        return false;
    }
    if (st != ProxyStatus::kOk) {
        LOG(WARNING) << "nvlink_proxy: daemon refused block "
                     << reinterpret_cast<void *>(block.base) << " ("
                     << block.size << " bytes): " << proxyStatusName(st) << " "
                     << reply
                     << "; transfers from/to it use the base transport";
        return false;
    }
    block.registered_epoch = epoch_.load();
    return true;
}

bool NvlinkProxyTransport::acquireCopyConnection(Connection &conn) {
    const uint64_t epoch = epoch_.load();
    if (epoch == 0) return false;
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
    if (!reusable || conn.epoch != epoch_.load()) {
        close(conn.fd);
        conn.fd = -1;
        return;
    }
    std::lock_guard<std::mutex> lock(pool_mu_);
    idle_.push_back(conn);
}

NvlinkProxyTransport::CopyOutcome NvlinkProxyTransport::copy(
    const std::vector<CopyEntry> &entries, size_t begin, size_t count,
    std::string &error) {
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
    // gave up on cannot land after the fallback transport rewrote the data.
    rh.budget_ms = static_cast<uint32_t>(timeout_ms_ / 2);
    payload.reserve(sizeof(rh) + count * sizeof(CopyEntry));
    payload.append(reinterpret_cast<const char *>(&rh), sizeof(rh));
    payload.append(reinterpret_cast<const char *>(entries.data() + begin),
                   count * sizeof(CopyEntry));
    ProxyStatus st;
    std::string reply;
    if (!rpc(conn.fd, MsgType::kCopy, payload.data(), payload.size(), st,
             reply)) {
        error = std::string("daemon connection error: ") +
                (errno == EAGAIN || errno == EWOULDBLOCK ? "timed out"
                                                         : strerror(errno));
        releaseCopyConnection(conn, false);
        requestRecheck();
        return CopyOutcome::kNoDaemon;
    }
    releaseCopyConnection(conn, true);
    if (st != ProxyStatus::kOk) {
        error = proxyStatusName(st);
        if (reply.size() > sizeof(CopyResp)) {
            error += ": " + reply.substr(sizeof(CopyResp));
        }
        // Right after a daemon (re)start the peer may not have re-registered
        // its blocks yet: retry shortly instead of falling back.
        if ((st == ProxyStatus::kUnknownSource ||
             st == ProxyStatus::kUnknownDestination) &&
            nowNs() - epoch_since_ns_.load() <
                int64_t(reconnect_wait_ms_) * 1000000LL)
            return CopyOutcome::kRetryLater;
        return CopyOutcome::kFailed;
    }
    return CopyOutcome::kOk;
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
    char line[1024];
    snprintf(
        line, sizeof(line),
        "healthy=%d proxied_requests=%" PRIu64 " proxied_bytes=%" PRIu64
        " proxied_batches=%" PRIu64 " avg_batch_us=%" PRIu64
        " max_batch_us=%" PRIu64 " held_requests=%" PRIu64
        " max_hold_ms=%" PRIu64 " retried_requests=%" PRIu64
        " fallback_requests=%" PRIu64 " (daemon_unavailable=%" PRIu64
        " daemon_error=%" PRIu64 " not_servable=%" PRIu64
        ") fallback_retries=%" PRIu64 " failed_requests=%" PRIu64
        " remote_node_requests=%" PRIu64
        " registered_blocks=%zu/%zu connects=%" PRIu64,
        healthy_.load() ? 1 : 0, counters_.proxy_requests.load(),
        counters_.proxy_bytes.load(), batches,
        batches ? counters_.proxy_us_total.load() / batches : 0,
        counters_.proxy_us_max.load(), counters_.held_requests.load(),
        counters_.held_us_max.load() / 1000, counters_.retried_requests.load(),
        counters_.fallback_requests.load(), counters_.fallback_unhealthy.load(),
        counters_.fallback_daemon_error.load(),
        counters_.fallback_untranslatable.load(),
        counters_.fallback_retries.load(), counters_.failed_requests.load(),
        counters_.remote_node_requests.load(), registered, blocks,
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
    const int64_t abandon_ns =
        deadline_ns + std::max<int64_t>(timeout_ms_, 10000) * 1000000LL;
    auto res = fallback_submitter_->run(
        fallback_, requests, deadline_ns, abandon_ns, [&](size_t i, bool ok) {
            ok ? completeTask(tasks[i]) : failTask(tasks[i]);
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

    std::vector<TransferTask *> pending(task_list.begin(), task_list.end());
    std::vector<TransferTask *> not_servable, failed, unavailable;
    std::string error;
    bool held = false;
    int round = 0;
    while (!pending.empty()) {
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
        std::vector<TransferTask *> proxied, retry;
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
            const CopyOutcome outcome = copy(entries, begin, count, err);
            if (outcome != CopyOutcome::kOk) {
                auto &dst = outcome == CopyOutcome::kFailed ? failed : retry;
                dst.insert(dst.end(), proxied.begin() + begin,
                           proxied.begin() + begin + count);
                retry_later |= outcome == CopyOutcome::kRetryLater;
                error = err;
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
        pending.swap(retry);
        if (pending.empty()) break;
        // Daemon restarted or a peer not re-registered yet: retry through
        // the proxy while the reconnect window lasts.
        const int64_t now = nowNs();
        if (now >= hold_deadline) {
            failed.insert(failed.end(), pending.begin(), pending.end());
            pending.clear();
            break;
        }
        counters_.retried_requests += pending.size();
        if (retry_later) {
            const int64_t backoff_ns =
                std::min<int64_t>(200, 10LL << std::min(round, 5)) * 1000000LL;
            std::this_thread::sleep_for(std::chrono::nanoseconds(
                std::min<int64_t>(backoff_ns, hold_deadline - now)));
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
    fallback(failed, FallbackReason::kDaemonError, error, deadline);
    return Status::OK();
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
    // Host memory, and device memory that cannot be exported through legacy
    // CUDA IPC, stays on the base transport only; that is not an error.
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, addr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    if (attr.type != cudaMemoryTypeDevice) return 0;
    CUdeviceptr base_ptr = 0;
    size_t alloc_size = 0;
    CUresult cu_err = cuMemGetAddressRange(&base_ptr, &alloc_size,
                                           reinterpret_cast<CUdeviceptr>(addr));
    if (cu_err != CUDA_SUCCESS) {
        LOG(WARNING) << "nvlink_proxy: cuMemGetAddressRange(" << addr
                     << ") failed (" << cu_err
                     << "); buffer stays on the base transport";
        return 0;
    }
    Block fresh;
    fresh.base = static_cast<uint64_t>(base_ptr);
    fresh.size = alloc_size;
    if (!deviceUuid(attr.device, fresh.uuid)) return 0;
    cudaIpcMemHandle_t handle;
    err = cudaIpcGetMemHandle(&handle, reinterpret_cast<void *>(base_ptr));
    if (err != cudaSuccess) {
        cudaGetLastError();
        LOG(WARNING) << "nvlink_proxy: buffer " << addr
                     << " cannot be exported with cudaIpcGetMemHandle ("
                     << cudaGetErrorString(err)
                     << "); it stays on the base transport";
        return 0;
    }
    memcpy(fresh.handle, &handle, kIpcHandleSize);

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
