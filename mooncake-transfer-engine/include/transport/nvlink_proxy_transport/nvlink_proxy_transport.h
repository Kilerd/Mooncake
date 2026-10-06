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

#ifndef NVLINK_PROXY_TRANSPORT_H_
#define NVLINK_PROXY_TRANSPORT_H_

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "transfer_metadata.h"
#include "transport/nvlink_proxy_transport/nvlink_proxy_protocol.h"
#include "transport/transport.h"

namespace mooncake {

// Same-node GPU-to-GPU transport for engines whose containers each see only
// their own GPU. Copies are executed by a node-local daemon
// (mooncake_nvlink_proxy) that sees every GPU of the node.
//
// The transport is additive: it is installed next to the base transport
// (rdma or tcp) when MC_NVLINK_PROXY_SOCKET is set. Device buffers are
// published under both protocols; MultiTransport routes a request here only
// when the target buffer was published on the same MC_NODE_ID, the local
// buffer is registered with the daemon and the daemon is healthy. Any proxy
// failure falls back to the base transport for that request.
class NvlinkProxyTransport : public Transport {
   public:
    NvlinkProxyTransport();
    ~NvlinkProxyTransport() override;

    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest> &entries) override;

    Status submitTransferTask(
        const std::vector<TransferTask *> &task_list) override;

    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus &status) override;

    // Transport used for requests the proxy cannot serve.
    void setFallbackTransport(Transport *transport) { fallback_ = transport; }

    // Routing gate used by MultiTransport::selectTransport for a target
    // buffer published under "nvlink_proxy".
    bool canRoute(const TransferRequest &request,
                  const BufferDesc &target_buffer);

    const std::string &nodeId() const { return node_id_; }

   protected:
    int install(std::string &local_server_name,
                std::shared_ptr<TransferMetadata> meta,
                std::shared_ptr<Topology> topo) override;

    int registerLocalMemory(void *addr, size_t length,
                            const std::string &location, bool remote_accessible,
                            bool update_metadata = true) override;

    int unregisterLocalMemory(void *addr, bool update_metadata = true) override;

    int registerLocalMemoryBatch(const std::vector<BufferEntry> &buffer_list,
                                 const std::string &location) override;

    int unregisterLocalMemoryBatch(
        const std::vector<void *> &addr_list) override;

    const char *getName() const override { return "nvlink_proxy"; }

   private:
    struct Block {
        uint64_t base = 0;
        uint64_t size = 0;
        uint8_t uuid[nvlink_proxy::kUuidSize] = {};
        uint8_t handle[nvlink_proxy::kIpcHandleSize] = {};
        size_t users = 0;               // registered buffers inside the block
        uint64_t registered_epoch = 0;  // daemon epoch it is registered in
    };

    struct Connection {
        int fd = -1;
        uint64_t epoch = 0;
    };

    enum class FallbackReason { kUnhealthy, kDaemonError, kUntranslatable };

    // Daemon I/O.
    int connectSocket(int timeout_ms);
    bool rpc(int fd, nvlink_proxy::MsgType type, const void *payload,
             size_t len, nvlink_proxy::ProxyStatus &status, std::string &reply);
    bool hello(int fd, bool control, uint64_t &epoch);
    bool ensureControlLocked();
    void closeControlLocked();
    bool registerBlockLocked(Block &block);
    bool acquireCopyConnection(Connection &conn);
    void releaseCopyConnection(Connection &conn, bool reusable);
    bool copy(const std::vector<nvlink_proxy::CopyEntry> &entries, size_t begin,
              size_t count, std::string &error);

    // Request translation: local address + remote buffer -> copy entry.
    bool findLocalBlock(uint64_t addr, uint64_t length, uint64_t &base,
                        bool require_registered);
    bool translate(const TransferRequest &request,
                   nvlink_proxy::CopyEntry &entry);
    const BufferDesc *findTargetBuffer(const SegmentDesc &desc, uint64_t addr,
                                       uint64_t length);

    void completeTask(TransferTask *task);
    Status fallback(const std::vector<TransferTask *> &tasks,
                    FallbackReason reason, const std::string &detail);
    void healthLoop();
    void requestRecheck();
    void maybeLogStats(bool force);

    std::string socket_path_;
    std::string node_id_;
    uint64_t client_id_ = 0;
    int timeout_ms_ = 30000;
    int stats_interval_s_ = 60;
    Transport *fallback_ = nullptr;

    // Local cudaMalloc blocks (base -> block) and registered buffers
    // (buffer addr -> block base).
    std::mutex blocks_mu_;
    std::map<uint64_t, Block> blocks_;
    std::unordered_map<uint64_t, uint64_t> buffers_;

    // Control connection: HELLO(control) + REGISTER/UNREGISTER/PING.
    std::mutex control_mu_;
    int control_fd_ = -1;
    std::atomic<uint64_t> epoch_{0};  // 0 = daemon not reachable
    std::atomic<bool> healthy_{false};

    // Pool of copy connections (one in-flight COPY each).
    std::mutex pool_mu_;
    std::vector<Connection> idle_;

    std::thread health_thread_;
    std::mutex health_mu_;
    std::condition_variable health_cv_;
    bool stop_ = false;
    bool recheck_ = false;  // a copy failed: re-validate the daemon now

    struct Counters {
        std::atomic<uint64_t> proxy_requests{0};
        std::atomic<uint64_t> proxy_bytes{0};
        std::atomic<uint64_t> proxy_batches{0};
        std::atomic<uint64_t> proxy_us_total{0};
        std::atomic<uint64_t> proxy_us_max{0};
        std::atomic<uint64_t> fallback_requests{0};
        std::atomic<uint64_t> fallback_unhealthy{0};
        std::atomic<uint64_t> fallback_daemon_error{0};
        std::atomic<uint64_t> fallback_untranslatable{0};
        std::atomic<uint64_t> remote_node_requests{0};
        std::atomic<uint64_t> reconnects{0};
    } counters_;
    std::atomic<int64_t> last_fallback_log_ns_{0};
    std::atomic<uint64_t> suppressed_fallback_logs_{0};
    std::mutex stats_mu_;
    int64_t next_stats_ns_ = 0;
    std::string last_stats_;
};

}  // namespace mooncake

#endif  // NVLINK_PROXY_TRANSPORT_H_
