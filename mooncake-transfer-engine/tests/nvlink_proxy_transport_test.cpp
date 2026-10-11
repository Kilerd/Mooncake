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

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "transfer_engine.h"
#include "transport/nvlink_proxy_transport/bounded_submitter.h"
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

TEST(NvlinkProxyProtocol, BufferRefViewParse) {
    const std::string s = "nvlp1|node-b|AbC|10|ff";
    const char *node;
    size_t node_len;
    uint64_t client, base, size;
    ASSERT_TRUE(decodeBufferRef(s, node, node_len, client, base, size));
    EXPECT_EQ(std::string(node, node_len), "node-b");
    EXPECT_EQ(client, 0xabcULL);
    EXPECT_EQ(base, 0x10ULL);
    EXPECT_EQ(size, 0xffULL);
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

// The nvlink_proxy transport moves same-node traffic onto its base transport
// through BoundedSubmitter when the daemon is unavailable. Eight concurrent
// submitters sharing one submitter (as many engine threads share one
// transport) must complete every request byte-exactly while the number of
// requests in flight never exceeds the bound. Submitting the same amount at
// once without a bound overflows the TCP lane queue (1024 queued + 1024
// pending per peer by default) and fails requests.
TEST(NvlinkProxyTransport, BoundedSubmitterKeepsTcpWithinCapacity) {
    setenv("MC_FORCE_TCP", "1", 1);
    auto engine = std::make_unique<TransferEngine>(true);
    ASSERT_EQ(engine->init(P2PHANDSHAKE, "127.0.0.1", "", 0), 0);
    Transport *tcp = engine->getTransport("tcp");
    ASSERT_NE(tcp, nullptr);

    const size_t kThreads = 8, kEntry = 2560, kCount = 3000;
    const size_t kTotal = kThreads * kCount;
    std::vector<char> src(kEntry * kTotal), dst(kEntry * kTotal);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = static_cast<char>((i * 2654435761u) >> 13);
    ASSERT_EQ(engine->registerLocalMemory(src.data(), src.size(), "cpu:0"), 0);
    ASSERT_EQ(engine->registerLocalMemory(dst.data(), dst.size(), "cpu:0"), 0);
    auto handle = engine->openSegment(engine->getLocalIpAndPort());
    ASSERT_NE(handle, static_cast<Transport::SegmentHandle>(-1));

    // Permuted destination slots, like a page-granular KV cache.
    std::vector<size_t> perm(kTotal);
    for (size_t i = 0; i < kTotal; ++i) perm[i] = (i * 7919) % kTotal;
    std::vector<Transport::TransferRequest> reqs(kTotal);
    for (size_t i = 0; i < kTotal; ++i) {
        reqs[i].opcode = Transport::TransferRequest::WRITE;
        reqs[i].source = src.data() + i * kEntry;
        reqs[i].target_id = handle;
        reqs[i].target_offset =
            reinterpret_cast<uint64_t>(dst.data() + perm[i] * kEntry);
        reqs[i].length = kEntry;
    }
    auto now_ns = []() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };

    // 1. Bounded, 8 concurrent submitters.
    BoundedSubmitter submitter(512, 256);
    std::atomic<size_t> ok{0}, bad{0}, max_inflight{0};
    std::atomic<bool> running{true};
    std::thread sampler([&] {
        while (running) {
            size_t v = submitter.inflight();
            size_t cur = max_inflight.load();
            while (v > cur && !max_inflight.compare_exchange_weak(cur, v)) {
            }
            std::this_thread::yield();
        }
    });
    std::vector<std::thread> threads;
    for (size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            std::vector<const Transport::TransferRequest *> ptrs(kCount);
            for (size_t i = 0; i < kCount; ++i) ptrs[i] = &reqs[t * kCount + i];
            const int64_t now = now_ns();
            auto res = submitter.run(
                tcp, ptrs, now + 30000000000LL, now + 60000000000LL,
                [&](size_t, bool success) { (success ? ok : bad)++; });
            EXPECT_EQ(res.failed, 0u);
        });
    }
    for (auto &th : threads) th.join();
    running = false;
    sampler.join();
    EXPECT_EQ(ok.load(), kTotal);
    EXPECT_EQ(bad.load(), 0u);
    EXPECT_LE(max_inflight.load(), 512u);
    EXPECT_EQ(submitter.inflight(), 0u);
    size_t mismatched = 0;
    for (size_t i = 0; i < kTotal; ++i)
        mismatched += memcmp(dst.data() + perm[i] * kEntry,
                             src.data() + i * kEntry, kEntry) != 0;
    EXPECT_EQ(mismatched, 0u);
    LOG(INFO) << "bounded: " << ok.load() << " requests from " << kThreads
              << " threads, max in flight " << max_inflight.load();

    // 2. Unbounded: one batch of every request is larger than the TCP lane
    // queue plus its admission queue. TCP used to reject most of it with
    // queue-full (and could strand the rest); it now chains such a submit,
    // so the batch completes without the bounded submitter as well.
    {
        auto batch = engine->allocateBatchID(kTotal);
        ASSERT_TRUE(engine->submitTransfer(batch, reqs).ok());
        size_t failed = 0, completed = 0;
        std::vector<int> state(kTotal, 0);
        auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (failed + completed < kTotal &&
               std::chrono::steady_clock::now() < deadline) {
            for (size_t i = 0; i < kTotal; ++i) {
                if (state[i]) continue;
                Transport::TransferStatus st;
                ASSERT_TRUE(engine->getTransferStatus(batch, i, st).ok());
                if (st.s == Transport::TransferStatusEnum::COMPLETED) {
                    state[i] = 1;
                    ++completed;
                } else if (st.s == Transport::TransferStatusEnum::FAILED) {
                    state[i] = 2;
                    ++failed;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        LOG(INFO) << "unbounded: completed=" << completed
                  << " failed=" << failed
                  << " unresolved=" << kTotal - completed - failed;
        EXPECT_EQ(completed, kTotal);
        EXPECT_EQ(failed, 0u);
        // Unresolved work would still reference the batch; free it only
        // when everything is terminal.
        if (completed + failed == kTotal) (void)engine->freeBatchID(batch);
    }
    unsetenv("MC_FORCE_TCP");
}

// The multi-protocol route cache must stay exact: requests that alternate
// between buffers route correctly, and an address in a gap between two
// registered buffers is rejected even right after a cached hit next to it.
TEST(NvlinkProxyTransport, MultiProtocolRouteCacheIsExact) {
    setenv("MC_FORCE_TCP", "1", 1);
    setenv("MC_NVLINK_PROXY_SOCKET",
           "/tmp/mooncake-nvlink-proxy-unit-test-absent.sock", 1);
    auto engine = std::make_unique<TransferEngine>(true);
    ASSERT_EQ(engine->init(P2PHANDSHAKE, "127.0.0.1", "", 0), 0);

    const size_t kMiB = 1 << 20;
    std::vector<char> pool(4 * kMiB, 0), src(kMiB);
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<char>(i * 7);
    char *a = pool.data(), *b = pool.data() + 2 * kMiB;  // gap in between
    ASSERT_EQ(engine->registerLocalMemory(a, kMiB, "cpu:0"), 0);
    ASSERT_EQ(engine->registerLocalMemory(b, kMiB, "cpu:0"), 0);
    ASSERT_EQ(engine->registerLocalMemory(src.data(), kMiB, "cpu:0"), 0);
    auto handle = engine->openSegment(engine->getLocalIpAndPort());
    ASSERT_NE(handle, static_cast<Transport::SegmentHandle>(-1));

    auto request = [&](char *dst, size_t len) {
        Transport::TransferRequest req;
        req.opcode = Transport::TransferRequest::WRITE;
        req.source = src.data();
        req.target_id = handle;
        req.target_offset = reinterpret_cast<uint64_t>(dst);
        req.length = len;
        return req;
    };
    auto wait = [&](Transport::BatchID batch, size_t n) {
        Transport::TransferStatus status;
        for (size_t i = 0; i < n; ++i) {
            auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(30);
            do {
                EXPECT_TRUE(engine->getTransferStatus(batch, i, status).ok());
                if (status.s == Transport::TransferStatusEnum::COMPLETED ||
                    status.s == Transport::TransferStatusEnum::FAILED)
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } while (std::chrono::steady_clock::now() < deadline);
            EXPECT_EQ(status.s, Transport::TransferStatusEnum::COMPLETED);
        }
    };

    // Alternate between the two buffers so every request switches the cache.
    std::vector<Transport::TransferRequest> reqs;
    for (int i = 0; i < 8; ++i)
        reqs.push_back(request((i % 2 ? b : a) + i * 4096, 4096));
    auto batch = engine->allocateBatchID(reqs.size());
    ASSERT_TRUE(engine->submitTransfer(batch, reqs).ok());
    wait(batch, reqs.size());
    EXPECT_TRUE(engine->freeBatchID(batch).ok());
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(memcmp((i % 2 ? b : a) + i * 4096, src.data(), 4096), 0);

    // Cached hit in |a| followed by a request into the gap after it.
    batch = engine->allocateBatchID(1);
    ASSERT_TRUE(
        engine->submitTransfer(batch, {request(a + kMiB - 4096, 4096)}).ok());
    wait(batch, 1);
    EXPECT_TRUE(engine->freeBatchID(batch).ok());
    batch = engine->allocateBatchID(1);
    EXPECT_FALSE(engine->submitTransfer(batch, {request(a + kMiB, 4096)}).ok());
    EXPECT_FALSE(engine->submitTransfer(batch, {request(b - 4096, 4096)}).ok());
    EXPECT_TRUE(engine->freeBatchID(batch).ok());

    EXPECT_EQ(engine->unregisterLocalMemory(a), 0);
    EXPECT_EQ(engine->unregisterLocalMemory(b), 0);
    EXPECT_EQ(engine->unregisterLocalMemory(src.data()), 0);
    unsetenv("MC_NVLINK_PROXY_SOCKET");
    unsetenv("MC_FORCE_TCP");
}
#endif

namespace {

// A base transport whose transfers end only when the test says so (or on
// cancel, if it honours cancel), and that can leave a cut transfer
// unconfirmed for a while.
class ScriptedTransport : public Transport {
   public:
    bool honour_cancel = true;
    int confirm_after_ms = 0;  // after a cancel; < 0: never
    std::atomic<int> cancels{0};
    std::vector<Slice *> slices;
    std::mutex mu;

    Status submitTransfer(BatchID,
                          const std::vector<TransferRequest> &) override {
        return Status::NotImplemented("scripted");
    }
    Status submitTransferTask(
        const std::vector<TransferTask *> &task_list) override {
        std::lock_guard<std::mutex> lock(mu);
        for (auto *task : task_list) {
            Slice *slice = getSliceCache().allocate();
            slice->task = task;
            slice->length = task->request->length;
            slice->status = Slice::PENDING;
            task->slice_list.push_back(slice);
            __sync_fetch_and_add(&task->slice_count, 1);
            slices.push_back(slice);
        }
        return Status::OK();
    }
    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus &status) override {
        auto &task = toBatchDesc(batch_id).task_list[task_id];
        const uint64_t ok = task.success_slice_count,
                       bad = task.failed_slice_count;
        status.s = ok + bad < task.slice_count ? TransferStatusEnum::WAITING
                   : bad                       ? TransferStatusEnum::FAILED
                                               : TransferStatusEnum::COMPLETED;
        return Status::OK();
    }
    void onBatchCanceled(BatchID batch_id) override {
        ++cancels;
        if (!honour_cancel) return;
        auto counter = toBatchDesc(batch_id).control.unconfirmed();
        std::lock_guard<std::mutex> lock(mu);
        for (auto *slice : slices) {
            if (slice->status != Slice::PENDING) continue;
            if (slice->task->batch_id != batch_id) continue;
            // A cut write: unconfirmed until its "receiver" answers.
            counter->fetch_add(1);
            slice->markFailed();
            if (confirm_after_ms >= 0) {
                std::thread([counter, ms = confirm_after_ms] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
                    counter->fetch_sub(1);
                }).detach();
            }
        }
    }
    void completeAll() {
        std::lock_guard<std::mutex> lock(mu);
        for (auto *slice : slices)
            if (slice->status == Slice::PENDING) slice->markSuccess();
    }

   private:
    int registerLocalMemory(void *, size_t, const std::string &, bool,
                            bool) override {
        return 0;
    }
    int unregisterLocalMemory(void *, bool) override { return 0; }
    int registerLocalMemoryBatch(const std::vector<BufferEntry> &,
                                 const std::string &) override {
        return 0;
    }
    int unregisterLocalMemoryBatch(const std::vector<void *> &) override {
        return 0;
    }
    const char *getName() const override { return "scripted"; }
};

int64_t steadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

// A fallback attempt that outlives its deadline is canceled and then waited
// for until quiesced -- also for the cut transfers its "receiver" confirms
// late -- instead of being abandoned, and its requests are not retried.
TEST(NvlinkProxyTransport, BoundedSubmitterCancelsAndWaitsForQuiescence) {
    ScriptedTransport base;
    base.confirm_after_ms = 150;
    std::vector<Transport::TransferRequest> reqs(4);
    std::vector<const Transport::TransferRequest *> ptrs;
    for (auto &r : reqs) {
        r.length = 64;
        ptrs.push_back(&r);
    }
    BoundedSubmitter submitter(16, 16);
    const int64_t t0 = steadyNowNs();
    std::atomic<int> failed{0}, abandoned_batches{0};
    auto res = submitter.run(
        &base, ptrs, t0 + 50LL * 1000 * 1000, t0 + 10LL * 1000 * 1000 * 1000,
        [&](size_t, bool ok) { failed += !ok; },
        [&](Transport::BatchDesc *) { ++abandoned_batches; });
    const int64_t took_ms = (steadyNowNs() - t0) / 1000000;
    EXPECT_EQ(base.cancels.load(), 1);
    EXPECT_EQ(abandoned_batches.load(), 0);
    EXPECT_EQ(res.abandoned, 0u);
    EXPECT_EQ(failed.load(), 4);
    EXPECT_EQ(res.retried, 0u);
    // Canceled about 1 s after the deadline, then 150 ms to confirm.
    EXPECT_GE(took_ms, 1150);
    EXPECT_LT(took_ms, 5000);
}

// An attempt that is not quiesced by the abandon time is handed over (its
// caller then counts it unconfirmed) before its requests are reported
// failed; it is freed only once quiesced.
TEST(NvlinkProxyTransport, BoundedSubmitterHandsOverAnUnquiescedAttempt) {
    ScriptedTransport base;
    base.honour_cancel = false;
    std::vector<Transport::TransferRequest> reqs(3);
    std::vector<const Transport::TransferRequest *> ptrs;
    for (auto &r : reqs) {
        r.length = 64;
        ptrs.push_back(&r);
    }
    BoundedSubmitter submitter(16, 16);
    const int64_t t0 = steadyNowNs();
    Transport::BatchDesc *handed = nullptr;
    bool handed_before_failure = true;
    int failed = 0;
    auto res = submitter.run(
        &base, ptrs, t0 + 20LL * 1000 * 1000, t0 + 200LL * 1000 * 1000,
        [&](size_t, bool ok) {
            if (!ok) {
                ++failed;
                if (!handed) handed_before_failure = false;
            }
        },
        [&](Transport::BatchDesc *batch) { handed = batch; });
    ASSERT_NE(handed, nullptr);
    EXPECT_TRUE(handed_before_failure);
    EXPECT_EQ(failed, 3);
    EXPECT_EQ(res.abandoned, 3u);
    EXPECT_FALSE(BoundedSubmitter::quiesced(*handed));
    base.completeAll();
    EXPECT_TRUE(BoundedSubmitter::quiesced(*handed));
    delete handed;
}
