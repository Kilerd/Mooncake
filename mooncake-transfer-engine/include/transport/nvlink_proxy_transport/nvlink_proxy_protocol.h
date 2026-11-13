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
    kRelease = 7,
    kRegisterFd = 8,
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
    // COPY not served yet: its GPU pair has not passed the daemon's P2P
    // self-test, and a test of the pair is queued or running. Nothing of the
    // copy was issued; the engine may send it again, and once the test
    // decided the daemon serves it or refuses it (NO_PEER_ACCESS). A pair
    // that is refused for any other reason, or untested with no test coming,
    // gets NO_PEER_ACCESS. Status values only grow, so no version bump: an
    // engine that does not know this one takes it as a failed copy and uses
    // its base transport, as it did for NO_PEER_ACCESS; a daemon that does
    // not know it never sends it.
    kPairUnderTest = 11,
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
        case ProxyStatus::kPairUnderTest:
            return "PAIR_UNDER_TEST";
    }
    return "UNKNOWN_STATUS";
}

// HELLO flag: this connection owns registrations. The daemon drops every
// registration made on it when the connection closes.
constexpr uint32_t kHelloFlagControl = 1u << 0;
// HELLO flag: node_id holds the node text (NUL-terminated, at most
// kHelloTokenOffset - 1 bytes) and, at kHelloTokenOffset, a random 64-bit
// token of the engine's process. The daemon tells processes apart by it:
// pid and node alone repeat across PID namespaces (containers of a pod).
// Engines of one process share it while any of them lives; one created when
// none is left draws a new one, so processes restored from one snapshot
// differ. Daemons that do not know the flag read only the text.
constexpr uint32_t kHelloFlagProcessToken = 1u << 1;
constexpr size_t kHelloTokenOffset = 56;

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
    char node_id[kNodeIdSize];  // NUL padded (see kHelloFlagProcessToken)
};
static_assert(kHelloTokenOffset + sizeof(uint64_t) <= kNodeIdSize,
              "process token fits in node_id");

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

// REGISTER_FD, on a control connection: REGISTER for a block of virtual
// memory management memory (cuMemCreate + cuMemMap), which has no legacy
// IPC handle. The message carries one file descriptor (SCM_RIGHTS, sent
// with the header) exported with cuMemExportToShareableHandle(
// CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR); the daemon imports it and maps
// [0, size) of the allocation (cuMemMap always maps from offset 0). The
// payload is a RegisterReq whose ipc_handle holds kVmmHandleTag and the
// allocation's CU_POINTER_ATTRIBUTE_BUFFER_ID: it tells a re-registration
// of the same allocation from a new one at the same address. Daemons that
// do not know the message reply BAD_REQUEST (the kernel discards the
// descriptor they do not receive).
constexpr char kVmmHandleTag[8] = {'M', 'C', 'V', 'M', 'M', 'F', 'D', '1'};

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

// RELEASE, on a control connection: the daemon drops every registration of
// that connection, then answers once none of the client's registrations is
// alive any more -- a copy in flight that resolved one may still map it --
// and so no mapping or import of its memory is left (OK), or when |wait_ms|
// passed (DEADLINE_EXCEEDED). An engine sends it before it disconnects for
// good (TransferEngine::shutdown), so that no other process imports the
// memory it exported once shutdown returned, e.g. before it is checkpointed.
// Any control connection of the client id may ask: one opened only for this
// drops nothing and waits for the registrations of an earlier connection.
// Daemons that do not know it reply BAD_REQUEST.
struct ReleaseReq {
    uint32_t wait_ms;
    uint32_t reserved;
};

struct ReleaseResp {
    uint64_t dropped;  // registrations dropped
    // Still held: registrations used by copies in flight, or (none left)
    // mappings and imports the daemon failed to release.
    uint64_t mappings;
};
#pragma pack(pop)

static_assert(sizeof(MsgHeader) == 24, "MsgHeader layout");
static_assert(sizeof(HelloReq) == 80, "HelloReq layout");
static_assert(sizeof(RegisterReq) == 96, "RegisterReq layout");
static_assert(sizeof(CopyEntry) == 56, "CopyEntry layout");
static_assert(sizeof(ReleaseReq) == 8, "ReleaseReq layout");
static_assert(sizeof(ReleaseResp) == 16, "ReleaseResp layout");

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

inline bool parseHex64(const char *p, size_t n, uint64_t &out) {
    if (n == 0 || n > 16) return false;
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i) {
        const char c = p[i];
        uint64_t d;
        if (c >= '0' && c <= '9')
            d = uint64_t(c - '0');
        else if (c >= 'a' && c <= 'f')
            d = uint64_t(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            d = uint64_t(c - 'A' + 10);
        else
            return false;
        v = (v << 4) | d;
    }
    out = v;
    return true;
}

// Allocation-free variant used on the per-request routing path: |node_id|
// points into |s|.
inline bool decodeBufferRef(const std::string &s, const char *&node_id,
                            size_t &node_id_len, uint64_t &client_id,
                            uint64_t &base, uint64_t &size) {
    static constexpr char kPrefix[] = "nvlp1|";
    constexpr size_t kPrefixLen = sizeof(kPrefix) - 1;
    if (s.size() <= kPrefixLen || s.compare(0, kPrefixLen, kPrefix) != 0)
        return false;
    // Split from the right: the node id never contains '|' (the client
    // sanitizes it), but parsing from the right keeps this robust.
    size_t p3 = s.rfind('|');
    if (p3 == std::string::npos || p3 < kPrefixLen) return false;
    size_t p2 = s.rfind('|', p3 - 1);
    if (p2 == std::string::npos || p2 < kPrefixLen) return false;
    size_t p1 = s.rfind('|', p2 - 1);
    if (p1 == std::string::npos || p1 < kPrefixLen) return false;
    node_id = s.data() + kPrefixLen;
    node_id_len = p1 - kPrefixLen;
    if (node_id_len == 0) return false;
    return parseHex64(s.data() + p1 + 1, p2 - p1 - 1, client_id) &&
           parseHex64(s.data() + p2 + 1, p3 - p2 - 1, base) &&
           parseHex64(s.data() + p3 + 1, s.size() - p3 - 1, size) &&
           client_id != 0 && size != 0;
}

inline bool decodeBufferRef(const std::string &s, BufferRef &ref) {
    const char *node;
    size_t node_len;
    if (!decodeBufferRef(s, node, node_len, ref.client_id, ref.base, ref.size))
        return false;
    ref.node_id.assign(node, node_len);
    return true;
}

}  // namespace nvlink_proxy
}  // namespace mooncake

#endif  // NVLINK_PROXY_PROTOCOL_H_
