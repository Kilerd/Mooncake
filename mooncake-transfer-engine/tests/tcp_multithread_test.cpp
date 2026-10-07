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
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
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
    while (true) {  // checks at least once, even with no time left
        if (!engine->getTransferStatus(batch, task, status).ok())
            return TransferStatusEnum::FAILED;
        if (status.s == TransferStatusEnum::COMPLETED ||
            status.s == TransferStatusEnum::FAILED)
            return status.s;
        if (std::chrono::steady_clock::now() >= deadline)
            return TransferStatusEnum::TIMEOUT;
        std::this_thread::yield();
    }
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

struct FanOutResult {
    size_t completed = 0;
    size_t failed = 0;
    size_t pending = 0;
    double seconds = 0;
};

// Submits `blocks` small, scattered blocks to one peer as a single batch and
// waits up to `timeout` for every task to become terminal.
FanOutResult runFanOut(const Peer& client, size_t blocks, size_t block_size,
                       std::chrono::seconds timeout) {
    std::vector<TransferRequest> requests;
    requests.reserve(blocks);
    for (size_t i = 0; i < blocks; ++i) {
        // A stride coprime to the block count scatters the destinations.
        const size_t dst = (i * 7919) % blocks;
        requests.push_back(makeRequest(client, TransferRequest::WRITE,
                                       client.base() + i * block_size,
                                       dst * block_size, block_size));
    }
    FanOutResult result;
    const auto start = std::chrono::steady_clock::now();
    auto batch = client.engine->allocateBatchID(blocks);
    EXPECT_TRUE(client.engine->submitTransfer(batch, requests).ok());
    const auto deadline = start + timeout;
    std::vector<bool> done(blocks, false);
    size_t remaining = blocks;
    while (remaining && std::chrono::steady_clock::now() < deadline) {
        for (size_t i = 0; i < blocks; ++i) {
            if (done[i]) continue;
            TransferStatus status;
            if (!client.engine->getTransferStatus(batch, i, status).ok())
                continue;
            if (status.s == TransferStatusEnum::COMPLETED) {
                ++result.completed;
            } else if (status.s == TransferStatusEnum::FAILED) {
                ++result.failed;
            } else {
                continue;
            }
            done[i] = true;
            --remaining;
        }
        if (remaining)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    result.pending = remaining;
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    if (!remaining) (void)client.engine->freeBatchID(batch);
    LOG(INFO) << "fan-out of " << blocks << " x " << block_size
              << " B: completed=" << result.completed
              << " failed=" << result.failed << " pending=" << result.pending
              << " in " << result.seconds << " s";
    return result;
}

// A fragmented transfer is many small blocks to one peer, far more than the
// per-peer lane queue (1024) plus admission queue (1024) hold. It must
// back-pressure and complete, not fail with queue-full or stall.
TEST(TcpMultiThreadTest, LargeFanOutToOnePeerCompletes) {
    constexpr size_t kBlocks = 20000;
    constexpr size_t kBlockSize = 4096;
    for (const char* io_threads : {"1", "4"}) {
        ScopedEnvVar io("MC_TCP_IO_THREADS", io_threads);
        Peer server;
        server.init(std::string("127.0.0.2:1836") + io_threads,
                    kBlocks * kBlockSize);
        ASSERT_TRUE(server.ok);
        Peer client;
        client.init(std::string("127.0.0.2:1837") + io_threads,
                    kBlocks * kBlockSize, &server);
        ASSERT_TRUE(client.ok);
        for (size_t i = 0; i < kBlocks; ++i)
            fillPattern(client.base() + i * kBlockSize, kBlockSize,
                        static_cast<uint32_t>(i));

        const auto result =
            runFanOut(client, kBlocks, kBlockSize, std::chrono::seconds(120));
        EXPECT_EQ(result.completed, kBlocks) << "io threads " << io_threads;
        EXPECT_EQ(result.failed, 0u) << "io threads " << io_threads;
        EXPECT_EQ(result.pending, 0u) << "io threads " << io_threads;
        size_t mismatched = 0;
        for (size_t i = 0; i < kBlocks; ++i) {
            const size_t dst = (i * 7919) % kBlocks;
            if (memcmp(client.base() + i * kBlockSize,
                       server.base() + dst * kBlockSize, kBlockSize) != 0)
                ++mismatched;
        }
        EXPECT_EQ(mismatched, 0u) << "io threads " << io_threads;
    }
}

// The same fan-out to a peer that has gone away fails every block promptly
// instead of retrying the dead peer once per block.
TEST(TcpMultiThreadTest, LargeFanOutToGonePeerFailsPromptly) {
    constexpr size_t kBlocks = 20000;
    constexpr size_t kBlockSize = 4096;
    auto server = std::make_unique<Peer>();
    server->init("127.0.0.2:18381", kBlocks * kBlockSize);
    ASSERT_TRUE(server->ok);
    Peer client;
    client.init("127.0.0.2:18382", kBlocks * kBlockSize, server.get());
    ASSERT_TRUE(client.ok);
    server.reset();

    const auto result =
        runFanOut(client, kBlocks, kBlockSize, std::chrono::seconds(60));
    EXPECT_EQ(result.failed, kBlocks);
    EXPECT_EQ(result.pending, 0u);
    EXPECT_LT(result.seconds, 20.0);
}

// Many concurrent submits, each small enough for the direct path, still
// overflow the bounded per-peer queue and some are rejected with
// queue-full. A rejection must not strand the work that was admitted: every
// admitted block completes and nothing stays pending.
TEST(TcpMultiThreadTest, QueueFullRejectionDoesNotStallAdmittedWork) {
    constexpr size_t kThreads = 8;
    constexpr size_t kBatches = 5;
    constexpr size_t kPerBatch = 100;  // well under the chaining threshold
    constexpr size_t kBlockSize = 4096;
    constexpr size_t kBlocks = kThreads * kBatches * kPerBatch;
    ScopedEnvVar queue("MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER", "256");
    ScopedEnvVar pending("MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER", "64");
    Peer server;
    server.init("127.0.0.2:18391", kBlocks * kBlockSize);
    ASSERT_TRUE(server.ok);
    Peer client;
    client.init("127.0.0.2:18392", kBlocks * kBlockSize, &server);
    ASSERT_TRUE(client.ok);

    std::atomic<size_t> completed{0}, failed{0}, pending_after{0};
    // One overall deadline: a stalled peer must show up as pending work,
    // not as a test that waits per task.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    std::vector<std::thread> threads;
    for (size_t t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            std::vector<Transport::BatchID> batches;
            for (size_t b = 0; b < kBatches; ++b) {
                std::vector<TransferRequest> requests;
                for (size_t i = 0; i < kPerBatch; ++i) {
                    const size_t block = (t * kBatches + b) * kPerBatch + i;
                    requests.push_back(
                        makeRequest(client, TransferRequest::WRITE,
                                    client.base() + block * kBlockSize,
                                    block * kBlockSize, kBlockSize));
                }
                auto batch = client.engine->allocateBatchID(kPerBatch);
                EXPECT_TRUE(
                    client.engine->submitTransfer(batch, requests).ok());
                batches.push_back(batch);
            }
            for (auto batch : batches) {
                for (size_t i = 0; i < kPerBatch; ++i) {
                    const auto left = std::max(
                        std::chrono::seconds(0),
                        std::chrono::duration_cast<std::chrono::seconds>(
                            deadline - std::chrono::steady_clock::now()));
                    switch (waitTask(client.engine.get(), batch, i, left)) {
                        case TransferStatusEnum::COMPLETED:
                            ++completed;
                            break;
                        case TransferStatusEnum::FAILED:
                            ++failed;
                            break;
                        default:
                            ++pending_after;
                    }
                }
                if (pending_after.load() == 0)
                    (void)client.engine->freeBatchID(batch);
            }
        });
    }
    for (auto& thread : threads) thread.join();
    LOG(INFO) << "concurrent overflow: completed=" << completed.load()
              << " failed=" << failed.load()
              << " pending=" << pending_after.load();
    EXPECT_EQ(pending_after.load(), 0u);
    EXPECT_GT(completed.load(), 0u);
    EXPECT_EQ(completed.load() + failed.load(), kBlocks);
}

namespace {
// Fails a test that hangs instead of letting it block the whole run: a
// transport hang must surface as a failure with a clear message.
class Watchdog : public ::testing::EmptyTestEventListener {
   public:
    explicit Watchdog(std::chrono::seconds limit) : limit_(limit) {
        thread_ = std::thread([this] { run(); });
    }
    ~Watchdog() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }

    void OnTestStart(const ::testing::TestInfo& info) override {
        std::lock_guard<std::mutex> lock(mutex_);
        name_ = std::string(info.test_suite_name()) + "." + info.name();
        deadline_ = std::chrono::steady_clock::now() + limit_;
        armed_ = true;
        cv_.notify_all();
    }

    void OnTestEnd(const ::testing::TestInfo&) override {
        std::lock_guard<std::mutex> lock(mutex_);
        armed_ = false;
        cv_.notify_all();
    }

   private:
    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stop_) {
            if (!armed_) {
                cv_.wait(lock);
                continue;
            }
            if (cv_.wait_until(lock, deadline_) == std::cv_status::timeout &&
                armed_ && std::chrono::steady_clock::now() >= deadline_) {
                fprintf(stderr, "watchdog: %s exceeded %llds, aborting\n",
                        name_.c_str(), (long long)limit_.count());
                fflush(stderr);
                std::abort();
            }
        }
    }

    std::chrono::seconds limit_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::string name_;
    std::chrono::steady_clock::time_point deadline_;
    bool armed_ = false;
    bool stop_ = false;
    std::thread thread_;
};
}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    // Every test here finishes in seconds; three minutes means a hang.
    ::testing::UnitTest::GetInstance()->listeners().Append(
        new Watchdog(std::chrono::seconds(180)));
    return RUN_ALL_TESTS();
}
