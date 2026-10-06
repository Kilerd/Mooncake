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

#ifndef NVLINK_PROXY_PROTOCOL_H_
#define NVLINK_PROXY_PROTOCOL_H_

// Wire protocol between the nvlink_proxy transport (client) and the
// node-local copy daemon (mooncake_nvlink_proxy).
//
// Containerized deployments often give every pod only its own GPU. Legacy
// CUDA IPC handles cannot be opened in such a container when the exporting
// GPU is not visible there, so two pods on one node cannot copy GPU to GPU
// directly. The daemon runs in a container that sees every GPU of the node:
// clients register their cudaMalloc blocks (64-byte legacy IPC handles sent
// as plain data) and ask the daemon to copy between registered blocks. The
// daemon opens both handles in the destination GPU's context with peer
// access enabled, so the copy runs over NVLink / PCIe P2P.
//
// Transport: AF_UNIX SOCK_STREAM, little-endian, packed structs. Every
// message is a MsgHeader followed by payload_len bytes. Replies echo the
// request type and seq and carry a ProxyStatus; a failed reply's payload is
// a human readable error (COPY replies prefix it with CopyResp).
//
// Registration keys are chosen by the client: (client_id, block base). The
// client publishes the key in its segment metadata, so the key stays valid
// when the daemon restarts; the client then re-registers the same keys.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace mooncake {
namespace nvlink_proxy {

constexpr uint32_t kMagic = 0x504e434d;  // "MCNP"
constexpr uint16_t kProtocolVersion = 1;
constexpr size_t kIpcHandleSize = 64;
constexpr size_t kUuidSize = 16;
constexpr size_t kNodeIdSize = 64;
constexpr uint32_t kMaxPayload = 64u << 20;
constexpr uint32_t kMaxCopyEntries = 1u << 20;

enum class MsgType : uint16_t {
    kHello = 1,
    kRegister = 2,
    kUnregister = 3,
    kCopy = 4,
    kPing = 5,
    kStats = 6,
};

enum class ProxyStatus : uint32_t {
    kOk = 0,
    kBadRequest = 1,
    kVersionMismatch = 2,
    kUnknownDevice = 3,
    kUnknownSource = 4,
    kUnknownDestination = 5,
    kOutOfRange = 6,
    kNoPeerAccess = 7,
    kCudaError = 8,
    kDeadlineExceeded = 9,
    kNotControlConnection = 10,
};

inline const char *proxyStatusName(ProxyStatus s) {
    switch (s) {
        case ProxyStatus::kOk:
            return "OK";
        case ProxyStatus::kBadRequest:
            return "BAD_REQUEST";
        case ProxyStatus::kVersionMismatch:
            return "VERSION_MISMATCH";
        case ProxyStatus::kUnknownDevice:
            return "UNKNOWN_DEVICE";
        case ProxyStatus::kUnknownSource:
            return "UNKNOWN_SOURCE";
        case ProxyStatus::kUnknownDestination:
            return "UNKNOWN_DESTINATION";
        case ProxyStatus::kOutOfRange:
            return "OUT_OF_RANGE";
        case ProxyStatus::kNoPeerAccess:
            return "NO_PEER_ACCESS";
        case ProxyStatus::kCudaError:
            return "CUDA_ERROR";
        case ProxyStatus::kDeadlineExceeded:
            return "DEADLINE_EXCEEDED";
        case ProxyStatus::kNotControlConnection:
            return "NOT_CONTROL_CONNECTION";
    }
    return "UNKNOWN_STATUS";
}

// HELLO flag: this connection owns registrations. The daemon drops every
// registration made on it when the connection closes.
constexpr uint32_t kHelloFlagControl = 1u << 0;

#pragma pack(push, 1)
struct MsgHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t type;
    uint32_t status;  // ProxyStatus in replies, 0 in requests
    uint32_t payload_len;
    uint64_t seq;
};

struct HelloReq {
    uint64_t client_id;
    uint32_t pid;
    uint32_t flags;
    char node_id[kNodeIdSize];  // diagnostics only, NUL padded
};

struct HelloResp {
    uint64_t daemon_epoch;  // random per daemon start
    uint32_t num_devices;
    uint32_t reserved;
};

struct RegisterReq {
    uint64_t base;  // cudaMalloc block base in the client's address space
    uint64_t size;
    uint8_t gpu_uuid[kUuidSize];
    uint8_t ipc_handle[kIpcHandleSize];
};

struct UnregisterReq {
    uint64_t base;
};

struct CopyReqHeader {
    uint32_t count;
    // Start budget: the daemon refuses to start copies once this many ms
    // have passed since it read the request (0 = no limit). Keeps a copy that
    // the client already gave up on (and retried on another transport) from
    // landing late.
    uint32_t budget_ms;
};

// Block references are (client_id, base); offsets are relative to base.
struct CopyEntry {
    uint64_t src_client;
    uint64_t src_base;
    uint64_t src_offset;
    uint64_t dst_client;
    uint64_t dst_base;
    uint64_t dst_offset;
    uint64_t length;
};

struct CopyResp {
    uint32_t failed_index;  // first failing entry, or count when n/a
    uint32_t reserved;
    uint64_t copied_bytes;
};

struct PingResp {
    uint64_t daemon_epoch;
    uint64_t uptime_ms;
};
#pragma pack(pop)

static_assert(sizeof(MsgHeader) == 24, "MsgHeader layout");
static_assert(sizeof(HelloReq) == 80, "HelloReq layout");
static_assert(sizeof(RegisterReq) == 96, "RegisterReq layout");
static_assert(sizeof(CopyEntry) == 56, "CopyEntry layout");

// ---------------------------------------------------------------------------
// Segment metadata. A device buffer registered under nvlink_proxy is
// published as a BufferDesc with protocol "nvlink_proxy" whose shm_name
// carries the locality key and the daemon registration key:
//   "nvlp1|<node_id>|<client_id hex>|<block base hex>|<block size hex>"
// ---------------------------------------------------------------------------
struct BufferRef {
    std::string node_id;
    uint64_t client_id = 0;
    uint64_t base = 0;
    uint64_t size = 0;
};

inline std::string encodeBufferRef(const BufferRef &ref) {
    char tail[64];
    snprintf(tail, sizeof(tail), "|%llx|%llx|%llx",
             static_cast<unsigned long long>(ref.client_id),
             static_cast<unsigned long long>(ref.base),
             static_cast<unsigned long long>(ref.size));
    return "nvlp1|" + ref.node_id + tail;
}

inline bool parseHex64(const std::string &s, uint64_t &out) {
    if (s.empty() || s.size() > 16) return false;
    char *end = nullptr;
    unsigned long long v = strtoull(s.c_str(), &end, 16);
    if (!end || *end != '\0') return false;
    out = static_cast<uint64_t>(v);
    return true;
}

inline bool decodeBufferRef(const std::string &s, BufferRef &ref) {
    static const std::string kPrefix = "nvlp1|";
    if (s.compare(0, kPrefix.size(), kPrefix) != 0) return false;
    // Split from the right: the node id itself never contains '|' (the
    // client sanitizes it), but parsing from the right keeps this robust.
    size_t p3 = s.rfind('|');
    if (p3 == std::string::npos || p3 < kPrefix.size()) return false;
    size_t p2 = s.rfind('|', p3 - 1);
    if (p2 == std::string::npos || p2 < kPrefix.size()) return false;
    size_t p1 = s.rfind('|', p2 - 1);
    if (p1 == std::string::npos || p1 < kPrefix.size()) return false;
    ref.node_id = s.substr(kPrefix.size(), p1 - kPrefix.size());
    if (ref.node_id.empty()) return false;
    return parseHex64(s.substr(p1 + 1, p2 - p1 - 1), ref.client_id) &&
           parseHex64(s.substr(p2 + 1, p3 - p2 - 1), ref.base) &&
           parseHex64(s.substr(p3 + 1), ref.size) && ref.client_id != 0 &&
           ref.size != 0;
}

}  // namespace nvlink_proxy
}  // namespace mooncake

#endif  // NVLINK_PROXY_PROTOCOL_H_
