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
// running daemon (a stand-in answers on a unix socket where one is needed).
// GPU-to-GPU behaviour is covered by nvlink_proxy_e2e_test.py on real
// hardware.

#include <dirent.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "transfer_engine.h"
#include "transport/nvlink_proxy_transport/bounded_submitter.h"
#include "transport/nvlink_proxy_transport/nvlink_proxy_protocol.h"
#include "transport/nvlink_proxy_transport/nvlink_proxy_transport.h"
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
    // Every pending transfer is cut, and its "receiver" can never confirm
    // that (a TCP receiver older than protocol v3).
    void failAllUnprovable() {
        std::lock_guard<std::mutex> lock(mu);
        for (auto *slice : slices) {
            if (slice->status != Slice::PENDING) continue;
            auto &control = toBatchDesc(slice->task->batch_id).control;
            control.unconfirmed()->fetch_add(1);
            control.unprovable()->fetch_add(1);
            slice->markFailed();
        }
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

// An attempt that settles with a transfer its receiver can never confirm:
// the caller learns of it first, and its failed requests are reported at
// once and never retried (they may still be applied).
TEST(NvlinkProxyTransport, BoundedSubmitterNeverRetriesAnUnprovableAttempt) {
    ScriptedTransport base;
    std::vector<Transport::TransferRequest> reqs(3);
    std::vector<const Transport::TransferRequest *> ptrs;
    for (auto &r : reqs) {
        r.length = 64;
        ptrs.push_back(&r);
    }
    BoundedSubmitter submitter(16, 16);
    const int64_t t0 = steadyNowNs();
    std::thread cut([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        base.failAllUnprovable();
    });
    std::atomic<int> failed{0}, unprovable{0};
    bool told_first = true;
    auto res = submitter.run(
        &base, ptrs, t0 + 5LL * 1000 * 1000 * 1000,
        t0 + 10LL * 1000 * 1000 * 1000,
        [&](size_t, bool ok) {
            if (!ok) {
                ++failed;
                if (!unprovable) told_first = false;
            }
        },
        nullptr, nullptr, [&] { ++unprovable; });
    cut.join();
    const int64_t took_ms = (steadyNowNs() - t0) / 1000000;
    EXPECT_EQ(unprovable.load(), 1);
    EXPECT_TRUE(told_first);
    EXPECT_EQ(failed.load(), 3);
    EXPECT_EQ(res.failed, 3u);
    EXPECT_EQ(res.retried, 0u);
    EXPECT_EQ(res.abandoned, 0u);
    EXPECT_EQ(base.cancels.load(), 0);
    EXPECT_LT(took_ms, 1000);
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

namespace mooncake {
struct NvlinkProxyTransportTestAccess {
    static void setDaemon(NvlinkProxyTransport &t, uint64_t epoch,
                          int64_t answered_ns) {
        t.epoch_ = epoch;
        t.last_answer_ns_ = answered_ns;
    }
    static void addUnansweredCopy(
        NvlinkProxyTransport &t, int fd, uint64_t epoch,
        const std::shared_ptr<std::atomic<int64_t>> &counter) {
        NvlinkProxyTransport::Unconfirmed u;
        u.fd = fd;
        u.epoch = epoch;
        u.parents.push_back(counter);
        t.addUnconfirmed(std::move(u));
    }
    static void reap(NvlinkProxyTransport &t) { t.reapUnconfirmed(); }
    // As if a daemon had accepted a block (no GPU memory in these tests).
    static void markRegistered(NvlinkProxyTransport &t) {
        std::lock_guard<std::mutex> lock(t.control_mu_);
        t.ever_registered_ = true;
    }
    // RELEASE of a transport that has no control connection.
    static bool releaseWithoutControl(NvlinkProxyTransport &t,
                                      const std::string &socket,
                                      uint64_t client) {
        t.socket_path_ = socket;
        t.client_id_ = client;
        t.release_wait_ms_ = 1000;
        std::lock_guard<std::mutex> lock(t.control_mu_);
        t.ever_registered_ = true;
        return t.releaseLocked();
    }
};
}  // namespace mooncake

// An unanswered COPY stays unconfirmed while its connection is open, even
// when a new daemon epoch is up (the old daemon may still be draining it);
// once the connection closed, a new epoch or an answer from the same daemon
// confirms it, and a late reply confirms it at once.
TEST(NvlinkProxyTransport, UnansweredCopyConfirmedOnlyAfterItsConnectionEnds) {
    using Access = NvlinkProxyTransportTestAccess;
    NvlinkProxyTransport proxy;
    auto counter = std::make_shared<std::atomic<int64_t>>(0);

    // A new daemon epoch while the old daemon still holds the connection.
    int fds[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
    Access::addUnansweredCopy(proxy, fds[0], 1, counter);
    EXPECT_EQ(counter->load(), 1);
    Access::setDaemon(proxy, 2, steadyNowNs());
    Access::reap(proxy);
    EXPECT_EQ(counter->load(), 1);
    close(fds[1]);  // the old daemon exits
    Access::reap(proxy);
    EXPECT_EQ(counter->load(), 0);

    // The same daemon closed it (no new epoch): confirmed only once that
    // daemon answers after the close; never while no daemon answers.
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
    Access::setDaemon(proxy, 3, steadyNowNs());
    Access::addUnansweredCopy(proxy, fds[0], 3, counter);
    close(fds[1]);
    Access::reap(proxy);
    EXPECT_EQ(counter->load(), 1);
    Access::setDaemon(proxy, 0, 0);
    Access::reap(proxy);
    EXPECT_EQ(counter->load(), 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Access::setDaemon(proxy, 3, steadyNowNs());
    Access::reap(proxy);
    EXPECT_EQ(counter->load(), 0);

    // The late reply.
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
    Access::addUnansweredCopy(proxy, fds[0], 3, counter);
    Access::reap(proxy);
    EXPECT_EQ(counter->load(), 1);
    const char reply = 1;
    ASSERT_EQ(write(fds[1], &reply, 1), 1);
    Access::reap(proxy);
    EXPECT_EQ(counter->load(), 0);
    close(fds[1]);
}

#if defined(USE_NVLINK_PROXY) && defined(USE_TCP)
namespace {

// A stand-in for mooncake_nvlink_proxy that answers HELLO, PING, REGISTER
// and RELEASE (with |release_status| after |release_delay_ms|) and records
// what the engines sent.
class FakeDaemon {
   public:
    struct Hello {
        uint64_t client = 0;
        uint64_t token = 0;
        bool control = false;
    };

    explicit FakeDaemon(ProxyStatus release_status = ProxyStatus::kOk,
                        int release_delay_ms = 0)
        : release_status_(release_status), release_delay_ms_(release_delay_ms) {
        path_ = "/tmp/mc-nvlp-fake-" + std::to_string(getpid()) + "-" +
                std::to_string(next_id_++) + ".sock";
        unlink(path_.c_str());
        listen_fd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        memcpy(addr.sun_path, path_.c_str(), path_.size() + 1);
        EXPECT_EQ(
            bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)),
            0);
        EXPECT_EQ(listen(listen_fd_, 16), 0);
        acceptor_ = std::thread([this] { acceptLoop(); });
    }

    ~FakeDaemon() {
        stopping_ = true;
        ::shutdown(listen_fd_, SHUT_RDWR);
        acceptor_.join();
        {
            std::lock_guard<std::mutex> lock(mu_);
            for (int fd : open_) ::shutdown(fd, SHUT_RDWR);
        }
        for (auto &t : serving_) t.join();
        close(listen_fd_);
        unlink(path_.c_str());
    }

    const std::string &path() const { return path_; }

    std::vector<Hello> hellos() {
        std::lock_guard<std::mutex> lock(mu_);
        return hellos_;
    }

    std::vector<uint64_t> releases() {
        std::lock_guard<std::mutex> lock(mu_);
        return releases_;
    }

    bool connected(uint64_t client) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto &kv : clients_)
            if (kv.second == client) return true;
        return false;
    }

    // Every engine connection was closed and its thread has exited.
    bool waitIdle(int timeout_ms) {
        const auto until = std::chrono::steady_clock::now() +
                           std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < until) {
            std::vector<std::thread> done;
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (open_.empty() && live_ == 0) done.swap(serving_);
            }
            if (!done.empty() || idle()) {
                for (auto &t : done) t.join();
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return false;
    }

   private:
    bool idle() {
        std::lock_guard<std::mutex> lock(mu_);
        return open_.empty() && live_ == 0;
    }

    void acceptLoop() {
        while (!stopping_) {
            const int fd = accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
            if (fd < 0) continue;
            std::lock_guard<std::mutex> lock(mu_);
            open_.insert(fd);
            ++live_;
            serving_.emplace_back([this, fd] { serve(fd); });
        }
    }

    static bool readAll(int fd, void *buf, size_t len) {
        char *p = static_cast<char *>(buf);
        while (len > 0) {
            const ssize_t n = recv(fd, p, len, 0);
            if (n <= 0) return false;
            p += n;
            len -= static_cast<size_t>(n);
        }
        return true;
    }

    static void reply(int fd, const MsgHeader &req, ProxyStatus st,
                      const void *payload, size_t len) {
        MsgHeader h = req;
        h.status = static_cast<uint32_t>(st);
        h.payload_len = static_cast<uint32_t>(len);
        std::string out(reinterpret_cast<const char *>(&h), sizeof(h));
        if (len) out.append(static_cast<const char *>(payload), len);
        (void)send(fd, out.data(), out.size(), MSG_NOSIGNAL);
    }

    void serve(int fd) {
        MsgHeader h;
        std::string payload;
        while (readAll(fd, &h, sizeof(h))) {
            payload.resize(h.payload_len);
            if (h.payload_len && !readAll(fd, &payload[0], h.payload_len))
                break;
            switch (static_cast<MsgType>(h.type)) {
                case MsgType::kHello: {
                    HelloReq req;
                    memcpy(&req, payload.data(), sizeof(req));
                    Hello rec;
                    rec.client = req.client_id;
                    rec.control = req.flags & kHelloFlagControl;
                    memcpy(&rec.token, req.node_id + kHelloTokenOffset,
                           sizeof(rec.token));
                    {
                        std::lock_guard<std::mutex> lock(mu_);
                        hellos_.push_back(rec);
                        clients_[fd] = rec.client;
                    }
                    HelloResp resp{};
                    resp.daemon_epoch = 0x5eed;
                    resp.num_devices = 1;
                    reply(fd, h, ProxyStatus::kOk, &resp, sizeof(resp));
                    break;
                }
                case MsgType::kPing: {
                    PingResp resp{};
                    resp.daemon_epoch = 0x5eed;
                    reply(fd, h, ProxyStatus::kOk, &resp, sizeof(resp));
                    break;
                }
                case MsgType::kRelease: {
                    {
                        std::lock_guard<std::mutex> lock(mu_);
                        releases_.push_back(clients_[fd]);
                    }
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(release_delay_ms_));
                    if (release_status_ == ProxyStatus::kBadRequest) {
                        const std::string msg = "unknown message type";
                        reply(fd, h, release_status_, msg.data(), msg.size());
                    } else {
                        ReleaseResp resp{};
                        resp.mappings =
                            release_status_ == ProxyStatus::kOk ? 0 : 1;
                        reply(fd, h, release_status_, &resp, sizeof(resp));
                    }
                    break;
                }
                default:
                    reply(fd, h, ProxyStatus::kOk, nullptr, 0);
            }
        }
        std::lock_guard<std::mutex> lock(mu_);
        close(fd);
        open_.erase(fd);
        clients_.erase(fd);
        --live_;
    }

    static std::atomic<int> next_id_;
    const ProxyStatus release_status_;
    const int release_delay_ms_;
    std::string path_;
    int listen_fd_ = -1;
    std::atomic<bool> stopping_{false};
    std::thread acceptor_;
    std::mutex mu_;
    std::vector<std::thread> serving_;
    std::set<int> open_;
    std::map<int, uint64_t> clients_;  // connection -> client id
    int live_ = 0;
    std::vector<Hello> hellos_;
    std::vector<uint64_t> releases_;
};
std::atomic<int> FakeDaemon::next_id_{0};

// What a checkpoint of the process would have to carry: sockets, epoll /
// eventfd descriptors, pipes and threads.
struct ProcessResources {
    int sockets = 0, anon_inodes = 0, pipes = 0, threads = 0;
};

ProcessResources processResources() {
    ProcessResources r;
    if (DIR *d = opendir("/proc/self/fd")) {
        while (dirent *e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            char link[256] = {};
            const std::string path = std::string("/proc/self/fd/") + e->d_name;
            if (readlink(path.c_str(), link, sizeof(link) - 1) <= 0) continue;
            const std::string target(link);
            if (target.rfind("socket:", 0) == 0) ++r.sockets;
            if (target.rfind("anon_inode:", 0) == 0) ++r.anon_inodes;
            if (target.rfind("pipe:", 0) == 0) ++r.pipes;
        }
        closedir(d);
    }
    if (DIR *d = opendir("/proc/self/task")) {
        while (dirent *e = readdir(d))
            if (e->d_name[0] != '.') ++r.threads;
        closedir(d);
    }
    return r;
}

bool operator==(const ProcessResources &a, const ProcessResources &b) {
    return a.sockets == b.sockets && a.anon_inodes == b.anon_inodes &&
           a.pipes == b.pipes && a.threads == b.threads;
}

// A joined thread is still listed for a moment: pthread_join returns when
// the kernel clears its tid, before the task is gone. Waits (at most 2 s)
// until the process holds |want|, else returns the last reading.
ProcessResources resourcesSettled(const ProcessResources &want) {
    const auto until =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    ProcessResources r = processResources();
    while (!(r == want) && std::chrono::steady_clock::now() < until) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        r = processResources();
    }
    return r;
}

// Two equal readings 50 ms apart.
ProcessResources resourcesStable() {
    ProcessResources r = processResources();
    for (int i = 0; i < 40; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const ProcessResources next = processResources();
        if (next == r) break;
        r = next;
    }
    return r;
}

// A TCP engine with nvlink_proxy installed against |daemon|, after one
// loopback write (lanes and server sessions exist).
std::unique_ptr<TransferEngine> busyEngine(FakeDaemon &daemon,
                                           NvlinkProxyTransport **proxy) {
    setenv("MC_FORCE_TCP", "1", 1);
    setenv("MC_NVLINK_PROXY_SOCKET", daemon.path().c_str(), 1);
    auto engine = std::make_unique<TransferEngine>(true);
    EXPECT_EQ(engine->init(P2PHANDSHAKE, "127.0.0.1", "", 0), 0);
    *proxy = dynamic_cast<NvlinkProxyTransport *>(
        engine->getTransport("nvlink_proxy"));
    EXPECT_NE(*proxy, nullptr);
    if (!*proxy) return engine;
    const uint64_t client = (*proxy)->clientId();
    const auto until =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!daemon.connected(client) &&
           std::chrono::steady_clock::now() < until)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_TRUE(daemon.connected(client));

    static std::vector<char> src(1 << 20), dst(1 << 20);
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<char>(i * 7);
    EXPECT_EQ(engine->registerLocalMemory(src.data(), src.size(), "cpu:0"), 0);
    EXPECT_EQ(engine->registerLocalMemory(dst.data(), dst.size(), "cpu:0"), 0);
    auto handle = engine->openSegment(engine->getLocalIpAndPort());
    auto batch = engine->allocateBatchID(1);
    Transport::TransferRequest req;
    req.opcode = Transport::TransferRequest::WRITE;
    req.source = src.data();
    req.target_id = handle;
    req.target_offset = reinterpret_cast<uint64_t>(dst.data());
    req.length = src.size();
    EXPECT_TRUE(engine->submitTransfer(batch, {req}).ok());
    Transport::TransferStatus status;
    do {
        EXPECT_TRUE(engine->getTransferStatus(batch, 0, status).ok());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (status.s != Transport::TransferStatusEnum::COMPLETED &&
             status.s != Transport::TransferStatusEnum::FAILED);
    EXPECT_EQ(status.s, Transport::TransferStatusEnum::COMPLETED);
    EXPECT_EQ(memcmp(src.data(), dst.data(), src.size()), 0);
    EXPECT_TRUE(engine->freeBatchID(batch).ok());
    return engine;
}

void clearEnv() {
    unsetenv("MC_NVLINK_PROXY_SOCKET");
    unsetenv("MC_FORCE_TCP");
}

}  // namespace

// shutdown() leaves nothing a checkpoint would carry: every socket (daemon
// connections, handshake listener, TCP acceptor, lanes and server sessions),
// epoll / eventfd descriptor and thread of the engine is gone when it
// returns, after the daemon confirmed the release; and it is quick.
TEST(NvlinkProxyShutdown, LeavesNoConnectionListenerOrThread) {
    using Access = NvlinkProxyTransportTestAccess;
    FakeDaemon daemon;
    NvlinkProxyTransport *proxy = nullptr;
    // A first engine sets up what lives as long as the process.
    {
        auto warm = busyEngine(daemon, &proxy);
        EXPECT_EQ(warm->shutdown(), 0);
    }
    ASSERT_TRUE(daemon.waitIdle(5000));
    const ProcessResources before = resourcesStable();

    auto engine = busyEngine(daemon, &proxy);
    ASSERT_NE(proxy, nullptr);
    const uint64_t client = proxy->clientId();
    const ProcessResources during = processResources();
    EXPECT_GT(during.sockets, before.sockets);
    EXPECT_GT(during.threads, before.threads);
    Access::markRegistered(*proxy);

    const int64_t t0 = steadyNowNs();
    EXPECT_EQ(engine->shutdown(), 0);
    const int64_t took_ms = (steadyNowNs() - t0) / 1000000;
    LOG(INFO) << "shutdown took " << took_ms << " ms";
    EXPECT_LT(took_ms, 2000);
    EXPECT_EQ(daemon.releases(), std::vector<uint64_t>{client});
    // The daemon saw every connection of the engine close.
    ASSERT_TRUE(daemon.waitIdle(2000));
    const ProcessResources after = resourcesSettled(before);
    EXPECT_EQ(after.sockets, before.sockets);
    EXPECT_EQ(after.anon_inodes, before.anon_inodes);
    EXPECT_EQ(after.pipes, before.pipes);
    EXPECT_EQ(after.threads, before.threads);
    EXPECT_EQ(engine->shutdown(), 0);  // idempotent
    engine.reset();
    clearEnv();
}

// A daemon that does not know RELEASE (older than the engine): shutdown
// still closes everything but cannot confirm the mappings are gone.
TEST(NvlinkProxyShutdown, OldDaemonLeavesTheReleaseUnconfirmed) {
    using Access = NvlinkProxyTransportTestAccess;
    FakeDaemon daemon(ProxyStatus::kBadRequest);
    NvlinkProxyTransport *proxy = nullptr;
    auto engine = busyEngine(daemon, &proxy);
    ASSERT_NE(proxy, nullptr);
    Access::markRegistered(*proxy);
    EXPECT_EQ(engine->shutdown(), -1);
    EXPECT_EQ(daemon.releases().size(), 1u);
    EXPECT_TRUE(daemon.waitIdle(2000));
    clearEnv();
}

// The daemon answers RELEASE only once its mappings are closed: shutdown
// waits for that answer; mappings still in use after the wait are reported.
TEST(NvlinkProxyShutdown, WaitsForTheDaemonToCloseItsMappings) {
    using Access = NvlinkProxyTransportTestAccess;
    {
        FakeDaemon daemon(ProxyStatus::kOk, 300);
        NvlinkProxyTransport *proxy = nullptr;
        auto engine = busyEngine(daemon, &proxy);
        ASSERT_NE(proxy, nullptr);
        Access::markRegistered(*proxy);
        const int64_t t0 = steadyNowNs();
        EXPECT_EQ(engine->shutdown(), 0);
        EXPECT_GE((steadyNowNs() - t0) / 1000000, 300);
    }
    {
        FakeDaemon daemon(ProxyStatus::kDeadlineExceeded);
        NvlinkProxyTransport *proxy = nullptr;
        auto engine = busyEngine(daemon, &proxy);
        ASSERT_NE(proxy, nullptr);
        Access::markRegistered(*proxy);
        EXPECT_EQ(engine->shutdown(), -1);
        EXPECT_TRUE(daemon.waitIdle(2000));
    }
    clearEnv();
}

// No block was ever accepted by a daemon: nothing of this engine can be
// mapped, no RELEASE is needed.
TEST(NvlinkProxyShutdown, WithoutRegisteredBlocksSendsNoRelease) {
    FakeDaemon daemon;
    NvlinkProxyTransport *proxy = nullptr;
    auto engine = busyEngine(daemon, &proxy);
    ASSERT_NE(proxy, nullptr);
    EXPECT_EQ(engine->shutdown(), 0);
    EXPECT_TRUE(daemon.releases().empty());
    EXPECT_TRUE(daemon.waitIdle(2000));
    clearEnv();
}

// Without a control connection RELEASE goes over one opened for it; no
// daemon listening at all means no daemon holds a mapping.
TEST(NvlinkProxyShutdown, ReleaseWithoutControlConnection) {
    using Access = NvlinkProxyTransportTestAccess;
    FakeDaemon daemon;
    {
        NvlinkProxyTransport proxy;
        EXPECT_TRUE(Access::releaseWithoutControl(proxy, daemon.path(), 0xabc));
        EXPECT_EQ(daemon.releases(), std::vector<uint64_t>{0xabc});
        const auto hellos = daemon.hellos();
        ASSERT_EQ(hellos.size(), 1u);
        EXPECT_TRUE(hellos[0].control);
        EXPECT_TRUE(daemon.waitIdle(2000));
    }
    {
        NvlinkProxyTransport proxy;
        EXPECT_TRUE(Access::releaseWithoutControl(
            proxy, "/tmp/mc-nvlp-nobody-listens.sock", 0xabd));
    }
}

// Every engine draws a new client id; engines of a process share the
// process token while one of them lives, and one created after all were
// shut down draws a new token (processes restored from one snapshot then
// differ).
TEST(NvlinkProxyShutdown, NewEngineDrawsNewIdentity) {
    FakeDaemon daemon;
    NvlinkProxyTransport *p1 = nullptr, *p2 = nullptr, *p3 = nullptr;
    auto e1 = busyEngine(daemon, &p1);
    auto e2 = busyEngine(daemon, &p2);
    ASSERT_TRUE(p1 && p2);
    const uint64_t c1 = p1->clientId(), c2 = p2->clientId();
    const uint64_t t1 = p1->processToken(), t2 = p2->processToken();
    EXPECT_NE(c1, c2);
    EXPECT_NE(t1, 0u);
    EXPECT_EQ(t1, t2);
    EXPECT_EQ(e1->shutdown(), 0);
    EXPECT_EQ(e2->shutdown(), 0);
    auto e3 = busyEngine(daemon, &p3);
    ASSERT_TRUE(p3);
    EXPECT_NE(p3->clientId(), c1);
    EXPECT_NE(p3->clientId(), c2);
    EXPECT_NE(p3->processToken(), t1);
    // What the daemon saw in HELLO.
    std::set<uint64_t> tokens;
    for (const auto &h : daemon.hellos())
        if (h.client == p3->clientId()) tokens.insert(h.token);
    EXPECT_EQ(tokens, std::set<uint64_t>{p3->processToken()});
    EXPECT_EQ(e3->shutdown(), 0);
    clearEnv();
}
#endif
