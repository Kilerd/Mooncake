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

// Tests for the multi-threaded TCP transport (MC_TCP_IO_THREADS). Every
// socket, timer and resolver belongs to exactly one io thread, lanes to one
// peer are spread over the threads, and accepted connections are spread over
// them on the serving side. These tests pin what must not change because of
// that: concurrent transfers stay byte-exact, the bytes of one transfer land
// in order, failures stay bounded, and shutdown with transfers in flight
// leaves every slice terminal exactly once.

#include <dirent.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "transfer_engine.h"
#include "transport/transport.h"

using namespace mooncake;

namespace {

class ScopedEnvVar {
   public:
    ScopedEnvVar(const char* name, const char* value) : name_(name) {
        if (const char* old = std::getenv(name)) {
            had_old_value_ = true;
            old_value_ = old;
        }
        setenv(name_.c_str(), value, 1);
    }

    ~ScopedEnvVar() {
        if (had_old_value_)
            setenv(name_.c_str(), old_value_.c_str(), 1);
        else
            unsetenv(name_.c_str());
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

   private:
    std::string name_;
    std::string old_value_;
    bool had_old_value_ = false;
};

size_t countProcessThreads() {
    size_t count = 0;
    DIR* dir = opendir("/proc/self/task");
    if (!dir) return 0;
    while (auto* entry = readdir(dir)) {
        if (entry->d_name[0] != '.') ++count;
    }
    closedir(dir);
    return count;
}

// One TransferEngine with a registered, zeroed host pool. When `peer` is
// given, segment_id/remote_base address that peer's pool; otherwise the
// engine's own pool (loopback).
struct Peer {
    std::unique_ptr<TransferEngine> engine;
    void* pool = nullptr;
    size_t pool_size = 0;
    Transport::SegmentID segment_id = 0;
    uint64_t remote_base = 0;
    bool ok = false;

    ~Peer() {
        engine.reset();
        free(pool);
    }

    void init(const std::string& server_name, size_t size,
              const Peer* peer = nullptr) {
        engine = std::make_unique<TransferEngine>(false);
        auto hp = parseHostNameWithPort(server_name);
        ASSERT_EQ(engine->init("P2PHANDSHAKE", server_name, hp.first.c_str(),
                               hp.second),
                  0);
        ASSERT_NE(engine->installTransport("tcp", nullptr), nullptr);
        pool_size = size;
        pool = aligned_alloc(4096, size);
        ASSERT_NE(pool, nullptr);
        memset(pool, 0, size);
        ASSERT_EQ(engine->registerLocalMemory(pool, size, "cpu:0"), 0);
        const std::string target = peer ? peer->engine->getLocalIpAndPort()
                                        : engine->getLocalIpAndPort();
        segment_id = engine->openSegment(target);
        auto desc = engine->getMetadata()->getSegmentDescByID(segment_id);
        ASSERT_NE(desc, nullptr);
        remote_base = (uint64_t)desc->buffers[0].addr;
        ok = true;
    }

    char* base() const { return static_cast<char*>(pool); }
};

TransferRequest makeRequest(const Peer& from, TransferRequest::OpCode opcode,
                            void* local, uint64_t remote_offset,
                            size_t length) {
    TransferRequest request;
    request.opcode = opcode;
    request.length = length;
    request.source = local;
    request.target_id = from.segment_id;
    request.target_offset = from.remote_base + remote_offset;
    return request;
}

TransferStatusEnum waitTask(TransferEngine* engine, Transport::BatchID batch,
                            size_t task, std::chrono::seconds timeout) {
    TransferStatus status;
    status.s = TransferStatusEnum::WAITING;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (!engine->getTransferStatus(batch, task, status).ok())
            return TransferStatusEnum::FAILED;
        if (status.s == TransferStatusEnum::COMPLETED ||
            status.s == TransferStatusEnum::FAILED)
            return status.s;
        std::this_thread::yield();
    }
    return TransferStatusEnum::TIMEOUT;
}

// Submit `requests` as one batch and wait for all of them.
bool runBatch(TransferEngine* engine,
              const std::vector<TransferRequest>& requests,
              std::chrono::seconds timeout = std::chrono::seconds(30)) {
    auto batch = engine->allocateBatchID(requests.size());
    if (!engine->submitTransfer(batch, requests).ok()) return false;
    bool all_ok = true;
    for (size_t i = 0; i < requests.size(); ++i)
        all_ok &= waitTask(engine, batch, i, timeout) ==
                  TransferStatusEnum::COMPLETED;
    (void)engine->freeBatchID(batch);
    return all_ok;
}

// Position- and generation-dependent bytes: a reordered, duplicated or
// missing 64 KiB chunk anywhere in a transfer changes the content.
void fillPattern(char* dst, size_t length, uint32_t seed) {
    auto* words = reinterpret_cast<uint32_t*>(dst);
    for (size_t i = 0; i < length / sizeof(uint32_t); ++i)
        words[i] = seed * 2654435761u + static_cast<uint32_t>(i);
}

void reclaimBatchAfterShutdown(Transport::BatchID batch) {
#ifndef CONFIG_USE_BATCH_DESC_SET
    delete &Transport::toBatchDesc(batch);
#else
    (void)batch;
#endif
}

void expectEverySliceTerminalOnce(Transport::BatchID batch) {
    const auto& desc = Transport::toBatchDesc(batch);
    for (size_t task_id = 0; task_id < desc.task_list.size(); ++task_id) {
        const auto& task = desc.task_list[task_id];
        const uint64_t success =
            __atomic_load_n(&task.success_slice_count, __ATOMIC_RELAXED);
        const uint64_t failed =
            __atomic_load_n(&task.failed_slice_count, __ATOMIC_RELAXED);
        const uint64_t slices =
            __atomic_load_n(&task.slice_count, __ATOMIC_RELAXED);
        EXPECT_EQ(success + failed, slices) << "task " << task_id;
    }
}

}  // namespace

// MC_TCP_IO_THREADS adds exactly that many io threads per transport.
TEST(TcpMultiThreadTest, IoThreadsSettingControlsWorkerCount) {
    auto threads_added = [](const char* io_threads, const char* name) {
        ScopedEnvVar env("MC_TCP_IO_THREADS", io_threads);
        auto engine = std::make_unique<TransferEngine>(false);
        auto hp = parseHostNameWithPort(name);
        EXPECT_EQ(
            engine->init("P2PHANDSHAKE", name, hp.first.c_str(), hp.second), 0);
        const size_t after_init = countProcessThreads();
        EXPECT_NE(engine->installTransport("tcp", nullptr), nullptr);
        const size_t after_install = countProcessThreads();
        engine.reset();
        return after_install - after_init;
    };
    const size_t one = threads_added("1", "127.0.0.2:18301");
    const size_t five = threads_added("5", "127.0.0.2:18302");
    EXPECT_EQ(five - one, 4u);
    // An invalid value falls back to the default instead of failing.
    const size_t invalid = threads_added("0", "127.0.0.2:18303");
    EXPECT_GE(invalid, one);
}

// Many submitters, each with its own region pair, write a fresh
// generation and read it back through the same lanes. Every read must
// return exactly the last completed write: per-transfer ordering and
// cross-thread isolation both hold with several io threads per side.
TEST(TcpMultiThreadTest, ConcurrentTransfersToOnePeerStayByteExact) {
    ScopedEnvVar io_threads("MC_TCP_IO_THREADS", "4");
    ScopedEnvVar lanes("MC_TCP_LANES_PER_PEER", "4");
    constexpr size_t kRegion = 8ull << 20;  // 128 chunks of 64 KiB
    constexpr int kSubmitters = 8;
    constexpr int kIterations = 12;

    Peer server;
    server.init("127.0.0.2:18311", kSubmitters * kRegion);
    ASSERT_TRUE(server.ok);
    Peer client;
    client.init("127.0.0.2:18312", 2 * kSubmitters * kRegion, &server);
    ASSERT_TRUE(client.ok);

    std::atomic<int> mismatches{0};
    std::atomic<int> failures{0};
    std::vector<std::thread> submitters;
    for (int t = 0; t < kSubmitters; ++t) {
        submitters.emplace_back([&, t] {
            char* src = client.base() + (2 * t) * kRegion;
            char* dst = client.base() + (2 * t + 1) * kRegion;
            const uint64_t remote = t * kRegion;
            for (int iter = 0; iter < kIterations; ++iter) {
                fillPattern(src, kRegion, (t << 16) | iter);
                // Split into requests of uneven size so several lanes carry
                // pieces of one generation at once.
                std::vector<TransferRequest> writes;
                const size_t piece = kRegion / 4 + 4096 * (iter % 3);
                for (size_t off = 0; off < kRegion; off += piece) {
                    const size_t len = std::min(piece, kRegion - off);
                    writes.push_back(makeRequest(client, TransferRequest::WRITE,
                                                 src + off, remote + off, len));
                }
                if (!runBatch(client.engine.get(), writes)) {
                    failures++;
                    continue;
                }
                memset(dst, 0, kRegion);
                if (!runBatch(client.engine.get(),
                              {makeRequest(client, TransferRequest::READ, dst,
                                           remote, kRegion)})) {
                    failures++;
                    continue;
                }
                if (memcmp(src, dst, kRegion) != 0 ||
                    memcmp(src, server.base() + remote, kRegion) != 0)
                    mismatches++;
            }
        });
    }
    for (auto& thread : submitters) thread.join();
    EXPECT_EQ(failures.load(), 0);
    EXPECT_EQ(mismatches.load(), 0);
}

// One large transfer is still one ordered byte stream on one connection;
// with more lanes than requests and several io threads the content must
// arrive intact for both directions.
TEST(TcpMultiThreadTest, LargeTransferArrivesInOrder) {
    ScopedEnvVar io_threads("MC_TCP_IO_THREADS", "8");
    ScopedEnvVar lanes("MC_TCP_LANES_PER_PEER", "8");
    constexpr size_t kLength = 96ull << 20;
    Peer server;
    server.init("127.0.0.2:18321", kLength);
    ASSERT_TRUE(server.ok);
    Peer client;
    client.init("127.0.0.2:18322", 2 * kLength, &server);
    ASSERT_TRUE(client.ok);

    fillPattern(client.base(), kLength, 0xC0FFEE);
    ASSERT_TRUE(runBatch(client.engine.get(),
                         {makeRequest(client, TransferRequest::WRITE,
                                      client.base(), 0, kLength)}));
    EXPECT_EQ(memcmp(client.base(), server.base(), kLength), 0);
    ASSERT_TRUE(runBatch(client.engine.get(),
                         {makeRequest(client, TransferRequest::READ,
                                      client.base() + kLength, 0, kLength)}));
    EXPECT_EQ(memcmp(client.base(), client.base() + kLength, kLength), 0);
}

// Failures stay failures and stay bounded with several io threads: an
// address outside every registered buffer is rejected, and a peer that has
// gone away fails new work instead of hanging it.
TEST(TcpMultiThreadTest, RejectedAndUnreachableTransfersFailPromptly) {
    ScopedEnvVar io_threads("MC_TCP_IO_THREADS", "4");
    constexpr size_t kPool = 4ull << 20;
    auto server = std::make_unique<Peer>();
    server->init("127.0.0.2:18331", kPool);
    ASSERT_TRUE(server->ok);
    Peer client;
    client.init("127.0.0.2:18332", kPool, server.get());
    ASSERT_TRUE(client.ok);

    // In range: completes.
    ASSERT_TRUE(runBatch(client.engine.get(),
                         {makeRequest(client, TransferRequest::WRITE,
                                      client.base(), 0, 64 * 1024)}));
    // Past the end of the peer's registered pool: rejected by the server.
    auto rejected = makeRequest(client, TransferRequest::WRITE, client.base(),
                                kPool, 64 * 1024);
    EXPECT_FALSE(
        runBatch(client.engine.get(), {rejected}, std::chrono::seconds(10)));

    // The peer goes away; new work must fail within the bound.
    server.reset();
    const auto start = std::chrono::steady_clock::now();
    std::vector<TransferRequest> reads;
    for (int i = 0; i < 8; ++i)
        reads.push_back(makeRequest(client, TransferRequest::READ,
                                    client.base() + i * 4096, i * 4096, 4096));
    auto batch = client.engine->allocateBatchID(reads.size());
    ASSERT_TRUE(client.engine->submitTransfer(batch, reads).ok());
    for (size_t i = 0; i < reads.size(); ++i)
        EXPECT_EQ(
            waitTask(client.engine.get(), batch, i, std::chrono::seconds(20)),
            TransferStatusEnum::FAILED)
            << "task " << i;
    EXPECT_LT(std::chrono::steady_clock::now() - start,
              std::chrono::seconds(20));
    (void)client.engine->freeBatchID(batch);
}

// Destroying the initiating engine while many transfers are in flight on
// several io threads must join cleanly and leave every slice terminal
// exactly once (completed or failed, never neither).
TEST(TcpMultiThreadTest, ShutdownWithTransfersInFlightTerminatesEverySlice) {
    ScopedEnvVar io_threads("MC_TCP_IO_THREADS", "4");
    ScopedEnvVar lanes("MC_TCP_LANES_PER_PEER", "4");
    constexpr size_t kPiece = 4ull << 20;
    constexpr size_t kPieces = 48;
    Peer server;
    server.init("127.0.0.2:18341", kPieces * kPiece);
    ASSERT_TRUE(server.ok);
    auto client = std::make_unique<Peer>();
    client->init("127.0.0.2:18342", kPieces * kPiece, &server);
    ASSERT_TRUE(client->ok);

    std::vector<TransferRequest> requests;
    for (size_t i = 0; i < kPieces; ++i)
        requests.push_back(makeRequest(
            *client, i % 2 ? TransferRequest::READ : TransferRequest::WRITE,
            client->base() + i * kPiece, i * kPiece, kPiece));
    auto batch = client->engine->allocateBatchID(requests.size());
    ASSERT_TRUE(client->engine->submitTransfer(batch, requests).ok());

    auto engine = std::move(client->engine);
    auto destruction =
        std::async(std::launch::async, [&engine] { engine.reset(); });
    ASSERT_EQ(destruction.wait_for(std::chrono::seconds(15)),
              std::future_status::ready);
    destruction.get();

    expectEverySliceTerminalOnce(batch);
    reclaimBatchAfterShutdown(batch);
}

// The serving side goes away mid-stream: in-flight and queued transfers on
// the initiator fail within the progress bound rather than hanging, and the
// initiator keeps working against a fresh peer afterwards.
TEST(TcpMultiThreadTest, PeerShutdownMidStreamFailsInFlightWork) {
    ScopedEnvVar io_threads("MC_TCP_IO_THREADS", "4");
    ScopedEnvVar progress("MC_TCP_PROGRESS_TIMEOUT_SEC", "5");
    constexpr size_t kPiece = 8ull << 20;
    constexpr size_t kPieces = 32;
    auto server = std::make_unique<Peer>();
    server->init("127.0.0.2:18351", kPieces * kPiece);
    ASSERT_TRUE(server->ok);
    Peer client;
    client.init("127.0.0.2:18352", kPieces * kPiece, server.get());
    ASSERT_TRUE(client.ok);

    std::vector<TransferRequest> requests;
    for (size_t i = 0; i < kPieces; ++i)
        requests.push_back(makeRequest(client, TransferRequest::WRITE,
                                       client.base() + i * kPiece, i * kPiece,
                                       kPiece));
    auto batch = client.engine->allocateBatchID(requests.size());
    ASSERT_TRUE(client.engine->submitTransfer(batch, requests).ok());
    server.reset();

    size_t terminal = 0;
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto status =
            waitTask(client.engine.get(), batch, i, std::chrono::seconds(30));
        if (status == TransferStatusEnum::COMPLETED ||
            status == TransferStatusEnum::FAILED)
            ++terminal;
    }
    EXPECT_EQ(terminal, requests.size());
    (void)client.engine->freeBatchID(batch);

    Peer fresh;
    fresh.init("127.0.0.2:18353", kPiece);
    ASSERT_TRUE(fresh.ok);
    auto segment =
        client.engine->openSegment(fresh.engine->getLocalIpAndPort());
    auto desc = client.engine->getMetadata()->getSegmentDescByID(segment);
    ASSERT_NE(desc, nullptr);
    TransferRequest write;
    write.opcode = TransferRequest::WRITE;
    write.length = kPiece;
    write.source = client.base();
    write.target_id = segment;
    write.target_offset = desc->buffers[0].addr;
    fillPattern(client.base(), kPiece, 7);
    EXPECT_TRUE(runBatch(client.engine.get(), {write}));
    EXPECT_EQ(memcmp(client.base(), fresh.base(), kPiece), 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    return RUN_ALL_TESTS();
}
