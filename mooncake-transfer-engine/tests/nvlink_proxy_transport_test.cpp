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

// Unit tests for the nvlink_proxy transport that need neither GPUs nor a
// running daemon. GPU-to-GPU behaviour is covered by
// nvlink_proxy_e2e_test.py on real hardware.

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "transfer_engine.h"
#include "transport/nvlink_proxy_transport/nvlink_proxy_protocol.h"
#include "transport/transport.h"

using namespace mooncake;
using namespace mooncake::nvlink_proxy;

TEST(NvlinkProxyProtocol, BufferRefRoundTrip) {
    BufferRef ref;
    ref.node_id = "node-a.example";
    ref.client_id = 0x0123456789abcdefULL;
    ref.base = 0x7f0000200000ULL;
    ref.size = 1ULL << 30;
    const std::string s = encodeBufferRef(ref);
    EXPECT_EQ(s, "nvlp1|node-a.example|123456789abcdef|7f0000200000|40000000");
    BufferRef out;
    ASSERT_TRUE(decodeBufferRef(s, out));
    EXPECT_EQ(out.node_id, ref.node_id);
    EXPECT_EQ(out.client_id, ref.client_id);
    EXPECT_EQ(out.base, ref.base);
    EXPECT_EQ(out.size, ref.size);
}

TEST(NvlinkProxyProtocol, BufferRefRejectsMalformed) {
    BufferRef out;
    EXPECT_FALSE(decodeBufferRef("", out));
    EXPECT_FALSE(decodeBufferRef("deadbeef", out));  // a legacy IPC handle
    EXPECT_FALSE(decodeBufferRef("nvlp1|node|1|2", out));
    EXPECT_FALSE(decodeBufferRef("nvlp1||1|2|3", out));
    EXPECT_FALSE(decodeBufferRef("nvlp1|node|0|2|3", out));    // client 0
    EXPECT_FALSE(decodeBufferRef("nvlp1|node|1|2|0", out));    // size 0
    EXPECT_FALSE(decodeBufferRef("nvlp1|node|xyz|2|3", out));  // not hex
    EXPECT_FALSE(
        decodeBufferRef("nvlp1|node|1|2|11111111111111111", out));  // >64b
    EXPECT_FALSE(decodeBufferRef("nvlp2|node|1|2|3", out));
}

TEST(NvlinkProxyProtocol, WireLayout) {
    EXPECT_EQ(sizeof(MsgHeader), 24u);
    EXPECT_EQ(sizeof(HelloReq), 80u);
    EXPECT_EQ(sizeof(HelloResp), 16u);
    EXPECT_EQ(sizeof(RegisterReq), 96u);
    EXPECT_EQ(sizeof(CopyReqHeader), 8u);
    EXPECT_EQ(sizeof(CopyEntry), 56u);
    EXPECT_EQ(sizeof(CopyResp), 16u);
}

#if defined(USE_NVLINK_PROXY) && defined(USE_TCP)
// With MC_NVLINK_PROXY_SOCKET pointing at a socket nobody listens on, the
// engine must behave like a TCP-only engine: initialization succeeds, host
// memory registers, and a loopback write lands through TCP.
TEST(NvlinkProxyTransport, NoDaemonUsesBaseTransport) {
    setenv("MC_FORCE_TCP", "1", 1);
    setenv("MC_NVLINK_PROXY_SOCKET",
           "/tmp/mooncake-nvlink-proxy-unit-test-absent.sock", 1);
    auto engine = std::make_unique<TransferEngine>(true);
    ASSERT_EQ(engine->init(P2PHANDSHAKE, "127.0.0.1", "", 0), 0);
    ASSERT_NE(engine->getTransport("tcp"), nullptr);
    ASSERT_NE(engine->getTransport("nvlink_proxy"), nullptr);

    const size_t kLen = 4 << 20;
    std::vector<char> src(kLen), dst(kLen, 0);
    for (size_t i = 0; i < kLen; ++i) src[i] = static_cast<char>(i * 131 + 7);
    ASSERT_EQ(engine->registerLocalMemory(src.data(), kLen, "cpu:0"), 0);
    ASSERT_EQ(engine->registerLocalMemory(dst.data(), kLen, "cpu:0"), 0);

    auto handle = engine->openSegment(engine->getLocalIpAndPort());
    ASSERT_NE(handle, static_cast<Transport::SegmentHandle>(-1));
    auto batch = engine->allocateBatchID(1);
    Transport::TransferRequest req;
    req.opcode = Transport::TransferRequest::WRITE;
    req.source = src.data();
    req.target_id = handle;
    req.target_offset = reinterpret_cast<uint64_t>(dst.data());
    req.length = kLen;
    ASSERT_TRUE(engine->submitTransfer(batch, {req}).ok());
    Transport::TransferStatus status;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    do {
        ASSERT_TRUE(engine->getTransferStatus(batch, 0, status).ok());
        if (status.s == Transport::TransferStatusEnum::COMPLETED ||
            status.s == Transport::TransferStatusEnum::FAILED)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    EXPECT_EQ(status.s, Transport::TransferStatusEnum::COMPLETED);
    EXPECT_EQ(memcmp(src.data(), dst.data(), kLen), 0);
    EXPECT_TRUE(engine->freeBatchID(batch).ok());
    EXPECT_EQ(engine->unregisterLocalMemory(src.data()), 0);
    EXPECT_EQ(engine->unregisterLocalMemory(dst.data()), 0);
    unsetenv("MC_NVLINK_PROXY_SOCKET");
    unsetenv("MC_FORCE_TCP");
}
#endif
