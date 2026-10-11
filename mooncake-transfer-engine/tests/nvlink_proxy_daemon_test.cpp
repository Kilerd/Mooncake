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

// Host-only parts of mooncake_nvlink_proxy that keep it safe to run next to
// engines that come and go: paced issue against the copy deadline, the IPC
// mapping table, the socket and the start-up checks (no GPU).

#include <gtest/gtest.h>
#include <pwd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "daemon_setup.h"
#include "ipc_imports.h"
#include "paced_issue.h"

using namespace mooncake::nvlink_proxy;

namespace {

uint64_t nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// A CPU stand-in for a GPU stream: a worker thread runs the queued
// operations in order; markers complete "events".
class FakeStream {
   public:
    FakeStream() : worker_([this]() { run(); }) {}
    ~FakeStream() {
        push(nullptr);
        worker_.join();
    }
    void push(std::function<void()> op) {
        {
            std::lock_guard<std::mutex> l(mu_);
            q_.push_back(std::move(op));
        }
        cv_.notify_all();
    }
    void sync() {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this]() { return q_.empty() && !busy_; });
    }
    size_t queued() {
        std::lock_guard<std::mutex> l(mu_);
        return q_.size() + (busy_ ? 1 : 0);
    }

   private:
    void run() {
        while (true) {
            std::function<void()> op;
            {
                std::unique_lock<std::mutex> l(mu_);
                cv_.wait(l, [this]() { return !q_.empty(); });
                op = std::move(q_.front());
                q_.pop_front();
                if (!op) return;
                busy_ = true;
            }
            op();
            {
                std::lock_guard<std::mutex> l(mu_);
                busy_ = false;
            }
            cv_.notify_all();
        }
    }
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> q_;
    bool busy_ = false;
    std::thread worker_;
};

struct FakeEvent {
    std::mutex mu;
    std::condition_variable cv;
    uint64_t recorded = 0, done = 0;
};

// Two lanes ("GPUs"); chunk k writes its index into its own slot of |dst|
// after |chunk_ms| and records when it was issued and when it landed.
struct PacedRig {
    std::map<int, std::unique_ptr<FakeStream>> streams;
    std::map<int, std::vector<std::unique_ptr<FakeEvent>>> events;
    std::vector<std::atomic<int>> dst;
    std::vector<uint64_t> landed_us;
    std::vector<uint64_t> issued_us;
    std::mutex landed_mu;
    std::atomic<size_t> max_outstanding{0};
    std::map<int, std::atomic<size_t>> outstanding;
    int chunk_ms;

    PacedRig(size_t chunks, int chunk_ms_)
        : dst(chunks), issued_us(chunks, 0), chunk_ms(chunk_ms_) {
        for (auto &d : dst) d = -1;
        for (int lane : {0, 1}) {
            streams[lane].reset(new FakeStream());
            for (size_t i = 0; i < kPacedWindow; ++i)
                events[lane].emplace_back(new FakeEvent());
            outstanding[lane] = 0;
        }
    }

    PacedResult run(const std::vector<PacedChunk> &chunks, uint64_t deadline_us,
                    int fail_at = -1) {
        return issuePaced(
            chunks, kPacedWindow, deadline_us, nowUs,
            [&](size_t k, size_t slot) -> int {
                if (int(k) == fail_at) return 700;
                const int lane = chunks[k].lane;
                const size_t now_out = ++outstanding[lane];
                size_t seen = max_outstanding.load();
                while (now_out > seen &&
                       !max_outstanding.compare_exchange_weak(seen, now_out)) {
                }
                issued_us[k] = nowUs();
                streams[lane]->push([this, k, lane]() {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(chunk_ms));
                    dst[k] = int(k);
                    {
                        std::lock_guard<std::mutex> l(landed_mu);
                        landed_us.push_back(nowUs());
                    }
                    --outstanding[lane];
                });
                FakeEvent *ev = events[lane][slot].get();
                uint64_t gen;
                {
                    std::lock_guard<std::mutex> l(ev->mu);
                    gen = ++ev->recorded;
                }
                streams[lane]->push([ev, gen]() {
                    {
                        std::lock_guard<std::mutex> l(ev->mu);
                        ev->done = gen;
                    }
                    ev->cv.notify_all();
                });
                return 0;
            },
            [&](int lane, size_t slot) -> int {
                FakeEvent *ev = events[lane][slot].get();
                std::unique_lock<std::mutex> l(ev->mu);
                const uint64_t gen = ev->recorded;
                ev->cv.wait(l, [&]() { return ev->done >= gen; });
                return 0;
            },
            [&](int lane) -> int {
                streams[lane]->sync();
                return 0;
            });
    }

    std::vector<int> snapshot() {
        std::vector<int> out;
        for (auto &d : dst) out.push_back(d.load());
        return out;
    }
    uint64_t lastLanded() {
        std::lock_guard<std::mutex> l(landed_mu);
        uint64_t t = 0;
        for (uint64_t x : landed_us) t = std::max(t, x);
        return t;
    }
};

// A model that has already measured |ms| per chunk of |bytes|.
void warmUp(RateModel &m, uint64_t bytes, int ms) {
    m.observe(bytes, ms * 1000.0 + RateModel::kOverheadUs);
}

std::vector<PacedChunk> uniform(size_t n, uint64_t bytes, RateModel *m,
                                bool two_lanes = false) {
    std::vector<PacedChunk> out;
    for (size_t k = 0; k < n; ++k)
        out.push_back({two_lanes ? int(k % 2) : 0, bytes, m});
    return out;
}

}  // namespace

TEST(NvlinkProxyPacedIssue, PlanChunksByEntriesAndBytes) {
    const std::vector<uint64_t> len = {10, 10, 10, 50, 10, 10, 200, 10};
    auto c = planChunks(len.data(), len.size(), 3, 40);
    ASSERT_EQ(c.size(), 5u);
    EXPECT_EQ(c[0].begin, 0u);  // 10 10 10 (entry limit)
    EXPECT_EQ(c[0].end, 3u);
    EXPECT_EQ(c[0].bytes, 30u);
    EXPECT_EQ(c[1].begin, 3u);  // 50 alone (over the byte limit)
    EXPECT_EQ(c[1].end, 4u);
    EXPECT_EQ(c[2].begin, 4u);  // 10 10
    EXPECT_EQ(c[2].end, 6u);
    EXPECT_EQ(c[3].begin, 6u);  // 200 alone
    EXPECT_EQ(c[3].end, 7u);
    EXPECT_EQ(c[4].bytes, 10u);
    EXPECT_TRUE(planChunks(len.data(), 0, 3, 40).empty());
    EXPECT_EQ(planChunks(len.data(), len.size(), 0, 0).size(), 1u);
    // Bytes only (gather-kernel rows): 20k rows of 2 KiB are one launch.
    const std::vector<uint64_t> rows(20000, 2048);
    EXPECT_EQ(planChunks(rows.data(), rows.size(), 0, 128ull << 20).size(), 1u);
    // A probe first: the first span at most 15 bytes, then the normal cap.
    c = planChunks(len.data(), len.size(), 0, 40, 15);
    ASSERT_GE(c.size(), 2u);
    EXPECT_EQ(c[0].end, 1u);
    EXPECT_EQ(c[1].begin, 1u);
    EXPECT_EQ(c[1].end, 3u);
    EXPECT_EQ(splitLength(100, 0), 1u);
    EXPECT_EQ(splitLength(100, 100), 1u);
    EXPECT_EQ(splitLength(101, 100), 2u);
    EXPECT_EQ(splitLength(1ull << 30, 32ull << 20), 32u);
}

TEST(NvlinkProxyPacedIssue, InterleavesLanes) {
    auto o = interleaveLanes({{3, 2}, {5, 4}});
    ASSERT_EQ(o.size(), 6u);
    const std::vector<std::pair<int, size_t>> want = {{3, 0}, {5, 0}, {3, 1},
                                                      {5, 1}, {5, 2}, {5, 3}};
    EXPECT_EQ(o, want);
}

TEST(NvlinkProxyPacedIssue, RateModelLearnsSlowdownsFast) {
    RateModel m;
    EXPECT_FALSE(m.warm());
    EXPECT_DOUBLE_EQ(m.psPerByte(), RateModel::kSeedPsPerByte);
    m.observe(1024, 5000.0);  // below kMinSampleBytes: pooled, not used yet
    EXPECT_FALSE(m.warm());
    EXPECT_DOUBLE_EQ(m.psPerByte(), RateModel::kSeedPsPerByte);
    const uint64_t big = RateModel::kFullWeightBytes;  // 8 MiB
    m.observe(big, 800.0 + RateModel::kOverheadUs);    // ~10 GB/s
    ASSERT_TRUE(m.warm());
    const double base = m.psPerByte();
    EXPECT_NEAR(base, 800e6 / double(big), 0.01);
    m.observe(big, 2400.0 + RateModel::kOverheadUs);  // 3x slower: half way
    EXPECT_NEAR(m.psPerByte(), base * 2, 0.01);
    const double slow = m.psPerByte();
    m.observe(big, 800.0 + RateModel::kOverheadUs);  // faster: an eighth
    EXPECT_NEAR(m.psPerByte(), slow - (slow - base) / 8, 0.01);
    EXPECT_NEAR(predictUs(big, base), RateModel::kOverheadUs + 800.0, 0.01);
}

// Once warm, chunks under kMinUpdateBytes do not move the estimate, and the
// others move it in proportion to their size: small, overhead-bound chunks
// cannot drag a measured rate down.
TEST(NvlinkProxyPacedIssue, RateModelWeighsSamplesByBytes) {
    RateModel m;
    const uint64_t big = RateModel::kFullWeightBytes;
    m.observe(big, 800.0 + RateModel::kOverheadUs);
    const double base = m.psPerByte();
    for (int i = 0; i < 100; ++i)
        m.observe(64 << 10, 500.0);  // 64 KiB in 500 us: ignored
    EXPECT_DOUBLE_EQ(m.psPerByte(), base);
    // 1 MiB, 3x slower: an eighth of the full-weight step (1/2).
    const uint64_t mib = RateModel::kMinUpdateBytes;
    const double slow_ps = 3 * base;
    m.observe(mib, slow_ps * double(mib) / 1e6 + RateModel::kOverheadUs);
    EXPECT_NEAR(m.psPerByte(), base + (slow_ps - base) / 16, 0.01);
}

// A model that only ever sees chunks smaller than kMinSampleBytes still
// warms up: the small samples are pooled into one (less their fixed costs),
// never slower than the seed (their fixed cost makes them look slow).
TEST(NvlinkProxyPacedIssue, RateModelWarmsFromSmallChunks) {
    RateModel m;
    const uint64_t small = 64ull << 10;
    const size_t n = RateModel::kMinSampleBytes / small;
    for (size_t i = 0; i + 1 < n; ++i) {
        m.observe(small, 32.0 + RateModel::kOverheadUs);  // ~2 GB/s
        EXPECT_FALSE(m.warm());
    }
    m.observe(small, 32.0 + RateModel::kOverheadUs);
    ASSERT_TRUE(m.warm());
    EXPECT_NEAR(m.psPerByte(), 32.0 * 1e6 / double(small), 0.01);
    // Pooled samples that are all fixed cost count as their whole time.
    RateModel f;
    for (size_t i = 0; i < n; ++i) f.observe(small, 1.0);
    ASSERT_TRUE(f.warm());
    EXPECT_NEAR(f.psPerByte(), double(n) * 1e6 / double(n * small), 1e-6);
    // Slow-looking small chunks: no lower than the seed.
    RateModel s;
    for (size_t i = 0; i < n; ++i) s.observe(small, 5000.0);
    ASSERT_TRUE(s.warm());
    EXPECT_DOUBLE_EQ(s.psPerByte(), RateModel::kSeedPsPerByte);
}

// A rate slower than the seed that nothing confirms expires: the model is
// cold again (seed rate, next chunk a probe). A fast one stays.
TEST(NvlinkProxyPacedIssue, RateModelForgetsAnUnconfirmedSlowRate) {
    RateModel slow(50 * 1000);  // 50 ms
    const uint64_t big = RateModel::kFullWeightBytes;
    slow.observe(big, 80000.0);  // ~0.1 GB/s, measured by a real chunk
    ASSERT_TRUE(slow.warm());
    EXPECT_GT(slow.psPerByte(), 5 * RateModel::kSeedPsPerByte);
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    EXPECT_FALSE(slow.warm());
    EXPECT_DOUBLE_EQ(slow.psPerByte(), RateModel::kSeedPsPerByte);
    slow.observe(big, 80000.0);  // measured again: slow again
    EXPECT_TRUE(slow.warm());
    EXPECT_GT(slow.psPerByte(), 5 * RateModel::kSeedPsPerByte);
    RateModel fast(50 * 1000);
    fast.observe(big, 800.0);
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    EXPECT_TRUE(fast.warm());
    EXPECT_LT(fast.psPerByte(), RateModel::kSeedPsPerByte);
}

// Every chunk issued and landed when it returns; never more than the window
// per lane in flight.
TEST(NvlinkProxyPacedIssue, IssuesEverythingWithinTheWindow) {
    constexpr size_t kChunks = 12;
    PacedRig rig(kChunks, 5);
    RateModel m;
    warmUp(m, 1 << 20, 5);
    const PacedResult r =
        rig.run(uniform(kChunks, 1 << 20, &m, true), UINT64_MAX);
    EXPECT_EQ(r.issued, kChunks);
    EXPECT_FALSE(r.deadline);
    EXPECT_EQ(r.error, 0);
    for (size_t k = 0; k < kChunks; ++k) EXPECT_EQ(rig.dst[k].load(), int(k));
    EXPECT_LE(rig.max_outstanding.load(), kPacedWindow);
}

// The deadline approaches mid-copy: a chunk is issued only if it and the one
// ahead of it finish in time at the measured rate; what is in flight lands
// before issuePaced returns (the reply), and nothing lands after the reply
// or the deadline.
TEST(NvlinkProxyPacedIssue, NoWriteLandsAfterTheDeadlineOrTheReply) {
    constexpr size_t kChunks = 40;
    constexpr int kChunkMs = 10;
    PacedRig rig(kChunks, kChunkMs);
    RateModel m;
    warmUp(m, 1 << 20, kChunkMs);
    const uint64_t deadline = nowUs() + 100 * 1000;
    const PacedResult r = rig.run(uniform(kChunks, 1 << 20, &m), deadline);
    const uint64_t reply = nowUs();
    EXPECT_TRUE(r.deadline);
    EXPECT_EQ(r.error, 0);
    EXPECT_LT(r.issued, kChunks);
    EXPECT_GE(r.issued, 5u);
    EXPECT_EQ(rig.streams[0]->queued(), 0u);
    EXPECT_LE(rig.lastLanded(), deadline + 3000)
        << "a write landed after the deadline";
    EXPECT_LE(reply, deadline + 3000);
    const std::vector<int> at_reply = rig.snapshot();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    EXPECT_EQ(rig.snapshot(), at_reply) << "a write landed after the reply";
    {
        std::lock_guard<std::mutex> l(rig.landed_mu);
        ASSERT_EQ(rig.landed_us.size(), r.issued);
        for (uint64_t t : rig.landed_us) EXPECT_LE(t, reply);
    }
    for (size_t k = 0; k < kChunks; ++k)
        EXPECT_EQ(at_reply[k], k < r.issued ? int(k) : -1) << k;
}

// A slow link with few long chunks (what a fixed margin misses): the
// measured rate keeps the second chunk from being issued at all.
TEST(NvlinkProxyPacedIssue, SlowChunksAreNotIssuedPastTheDeadline) {
    PacedRig rig(2, 80);  // two chunks of 80 ms
    RateModel m;
    warmUp(m, 64 << 20, 80);
    const uint64_t deadline = nowUs() + 120 * 1000;
    const PacedResult r = rig.run(uniform(2, 64 << 20, &m), deadline);
    EXPECT_TRUE(r.deadline);
    EXPECT_EQ(r.issued, 1u);  // 80 ms fits, 80 + 80 ms does not
    EXPECT_LE(rig.lastLanded(), deadline);
    // A copy that cannot finish is not started (a fresh model: the run
    // above has measured its chunk again).
    RateModel m2;
    warmUp(m2, 64 << 20, 80);
    PacedRig none(1, 80);
    const PacedResult n =
        none.run(uniform(1, 64 << 20, &m2), nowUs() + 50 * 1000);
    EXPECT_TRUE(n.deadline);
    EXPECT_EQ(n.issued, 0u);
    EXPECT_NEAR(n.predicted_us, 80000.0 + RateModel::kOverheadUs, 3000.0);
}

// A lane whose rate is not measured yet runs its first chunk alone; once
// that chunk has measured the rate, the window opens.
TEST(NvlinkProxyPacedIssue, ColdModelProbesAlone) {
    constexpr size_t kChunks = 6;
    PacedRig rig(kChunks, 20);
    RateModel m;  // cold, seed 1 GB/s: a 1 MiB chunk predicted ~1 ms
    const PacedResult r = rig.run(uniform(kChunks, 1 << 20, &m), UINT64_MAX);
    EXPECT_EQ(r.issued, kChunks);
    ASSERT_TRUE(m.warm());
    EXPECT_GT(m.psPerByte(), 10 * RateModel::kSeedPsPerByte);  // 20 ms/MiB
    EXPECT_GE(rig.issued_us[1], rig.issued_us[0] + 15 * 1000);
    EXPECT_LT(rig.issued_us[3], rig.issued_us[2] + 10 * 1000);
    EXPECT_LE(rig.max_outstanding.load(), kPacedWindow);
}

// The rate this copy actually achieves on a lane is a floor, even while the
// shared model (from earlier copies) is still optimistic.
TEST(NvlinkProxyPacedIssue, ThisCopysProgressOverridesAStaleModel) {
    constexpr size_t kChunks = 30;
    PacedRig rig(kChunks, 10);
    RateModel fast;
    warmUp(fast, 64 << 20, 1);  // measured 1 ms before; now 10 ms
    const uint64_t deadline = nowUs() + 100 * 1000;
    const PacedResult r = rig.run(uniform(kChunks, 64 << 20, &fast), deadline);
    EXPECT_TRUE(r.deadline);
    EXPECT_LE(rig.lastLanded(), deadline + 3000);
}

TEST(NvlinkProxyPacedIssue, ErrorStopsIssueAndStillDrains) {
    constexpr size_t kChunks = 10;
    PacedRig rig(kChunks, 5);
    RateModel m;
    warmUp(m, 1 << 20, 5);
    const PacedResult r =
        rig.run(uniform(kChunks, 1 << 20, &m, true), UINT64_MAX, 6);
    EXPECT_EQ(r.error, 700);
    EXPECT_EQ(r.issued, 6u);
    EXPECT_EQ(rig.streams[0]->queued() + rig.streams[1]->queued(), 0u);
    for (size_t k = 0; k < kChunks; ++k)
        EXPECT_EQ(rig.dst[k].load(), k < 6 ? int(k) : -1);
}

TEST(NvlinkProxyPacedIssue, DeadlineAlreadyPassedIssuesNothing) {
    PacedRig rig(4, 1);
    RateModel m;
    const PacedResult r = rig.run(uniform(4, 1 << 20, &m, true), 0);
    EXPECT_TRUE(r.deadline);
    EXPECT_EQ(r.issued, 0u);
}

namespace {

// Behaves like the driver (CUDA runtime API reference): opening a handle
// already open in a context returns the same pointer and counts the open;
// a close unmaps once every open was closed (|live| says which mappings
// still exist).
struct FakeDriver {
    std::map<std::pair<int, std::string>, std::pair<uintptr_t, int>> open_maps;
    std::map<uintptr_t, bool> live;
    uintptr_t next = 0x1000;
    int opens = 0, closes = 0;
    int fail_with = 0;

    int open(int dev, const uint8_t *handle, void **ptr) {
        if (fail_with) return fail_with;
        const auto key = std::make_pair(
            dev, std::string(reinterpret_cast<const char *>(handle),
                             kImportHandleBytes));
        auto it = open_maps.find(key);
        if (it == open_maps.end()) {
            it = open_maps.emplace(key, std::make_pair(next, 0)).first;
            live[next] = true;
            next += 0x1000;
        }
        ++it->second.second;
        ++opens;
        *ptr = reinterpret_cast<void *>(it->second.first);
        return 0;
    }
    int close(int, void *ptr) {
        const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        if (!live[p]) return 400;
        ++closes;
        for (auto it = open_maps.begin(); it != open_maps.end(); ++it) {
            if (it->second.first != p) continue;
            if (--it->second.second == 0) {
                live[p] = false;
                open_maps.erase(it);
            }
            break;
        }
        return 0;
    }
};

struct DriverRef {
    FakeDriver *d;
    int open(int dev, const uint8_t *h, void **p) { return d->open(dev, h, p); }
    int close(int dev, void *p) { return d->close(dev, p); }
};

std::vector<uint8_t> handleOf(const std::string &tag) {
    std::vector<uint8_t> h(kImportHandleBytes, 0);
    memcpy(h.data(), tag.data(), std::min(tag.size(), h.size()));
    return h;
}

using Table = ImportTable<DriverRef>;
const ImportOwner kProcA{101, "node-1"}, kProcB{202, "node-1"};

// Which clients still have a control connection.
struct Connected {
    std::set<uint64_t> *live;
    bool operator()(uint64_t c) const { return live->count(c) != 0; }
};

}  // namespace

// An engine disconnects: once its registrations are gone, every mapping it
// had (in every GPU context) is closed and nothing of it stays open.
TEST(NvlinkProxyImports, DisconnectClosesEveryMappingOfTheEngine) {
    FakeDriver drv;
    Table t(DriverRef{&drv});
    std::set<uint64_t> live = {0xA, 0xB};
    const Connected conn{&live};
    const uint64_t engine = 0xA, other = 0xB;
    const auto h1 = handleOf("kv-pool-1"), h2 = handleOf("kv-pool-2"),
               hb = handleOf("other-engine");
    // Two blocks, each mapped in its own GPU's context and the peer's.
    for (int dev : {0, 1}) {
        ASSERT_EQ(t.acquire(engine, kProcA, dev, h1.data(), conn).status,
                  Table::Status::kOk);
        ASSERT_EQ(t.acquire(engine, kProcA, dev, h2.data(), conn).status,
                  Table::Status::kOk);
    }
    ASSERT_EQ(t.acquire(other, kProcB, 1, hb.data(), conn).status,
              Table::Status::kOk);
    EXPECT_EQ(t.mappingsOf(engine), 4u);
    EXPECT_EQ(t.mappings(), 5u);
    live.erase(engine);
    for (int dev : {0, 1}) {
        EXPECT_TRUE(t.release(engine, dev, h1.data()));
        EXPECT_TRUE(t.release(engine, dev, h2.data()));
    }
    EXPECT_EQ(t.mappingsOf(engine), 0u);
    EXPECT_EQ(t.mappings(), 1u);  // the other engine's stays
    EXPECT_EQ(drv.closes, 4);
    EXPECT_EQ(drv.open_maps.size(), 1u);  // unmapped in the driver too
    EXPECT_EQ(t.opens(), 5u);
    EXPECT_EQ(t.closes(), 4u);
}

// The same engine registers a block again (reconnect) while a copy still
// holds the old registration: one mapping, open until both are gone.
TEST(NvlinkProxyImports, ReRegistrationSharesTheMapping) {
    FakeDriver drv;
    Table t(DriverRef{&drv});
    std::set<uint64_t> live = {0xA};
    const Connected conn{&live};
    const auto h = handleOf("kv-pool");
    auto a = t.acquire(0xA, kProcA, 0, h.data(), conn);
    auto b = t.acquire(0xA, kProcA, 0, h.data(), conn);
    ASSERT_EQ(b.status, Table::Status::kOk);
    EXPECT_TRUE(a.opened);
    EXPECT_FALSE(b.opened);
    EXPECT_EQ(a.ptr, b.ptr);
    EXPECT_EQ(drv.opens, 1);
    EXPECT_FALSE(t.release(0xA, 0, h.data()));  // the old registration
    EXPECT_TRUE(drv.live[reinterpret_cast<uintptr_t>(b.ptr)]);
    EXPECT_TRUE(t.release(0xA, 0, h.data()));  // the new one
    EXPECT_FALSE(drv.live[reinterpret_cast<uintptr_t>(b.ptr)]);
    EXPECT_FALSE(t.release(0xA, 0, h.data()));  // nothing left
}

// Two engines of one process (two client ids, same pid and node) register
// the same allocation: while the first is connected, the second shares its
// mapping; it stays open until both released it.
TEST(NvlinkProxyImports, SameProcessSharesWhileAHolderIsConnected) {
    FakeDriver drv;
    Table t(DriverRef{&drv});
    std::set<uint64_t> live = {0xA, 0xC};
    const Connected conn{&live};
    const auto h = handleOf("one-allocation");
    auto a = t.acquire(0xA, kProcA, 0, h.data(), conn);
    auto c = t.acquire(0xC, kProcA, 0, h.data(), conn);
    ASSERT_EQ(c.status, Table::Status::kOk);
    EXPECT_TRUE(c.shared);
    EXPECT_EQ(c.ptr, a.ptr);
    EXPECT_EQ(t.shares(), 1u);
    EXPECT_EQ(t.mappingsOf(0xC), 1u);
    live.erase(0xA);
    EXPECT_FALSE(t.release(0xA, 0, h.data()));  // 0xC still uses it
    EXPECT_TRUE(drv.live[reinterpret_cast<uintptr_t>(a.ptr)]);
    EXPECT_TRUE(t.release(0xC, 0, h.data()));
    EXPECT_FALSE(drv.live[reinterpret_cast<uintptr_t>(a.ptr)]);
}

// Another process -- even a connected one -- is never served from the
// mapping (its equal bytes cannot be told apart from another allocation).
TEST(NvlinkProxyImports, AnotherProcessIsRefusedEvenWhileConnected) {
    FakeDriver drv;
    Table t(DriverRef{&drv});
    std::set<uint64_t> live = {0xA, 0xB};
    const Connected conn{&live};
    const auto h = handleOf("same-bytes");
    ASSERT_EQ(t.acquire(0xA, kProcA, 0, h.data(), conn).status,
              Table::Status::kOk);
    auto b = t.acquire(0xB, kProcB, 0, h.data(), conn);
    EXPECT_EQ(b.status, Table::Status::kConflict);
    EXPECT_EQ(b.other, 0xAu);
    EXPECT_EQ(b.other_owner.pid, kProcA.pid);
    EXPECT_TRUE(b.other_connected);
    EXPECT_EQ(drv.opens, 1);
}

// A restarted engine (new client id, even reporting the same pid) whose
// handle bytes repeat its predecessor's while a copy of the disconnected
// predecessor still holds that mapping: never served from the old mapping;
// served from a fresh one once it is closed.
TEST(NvlinkProxyImports, RestartedEngineIsNeverServedFromAStaleMapping) {
    FakeDriver drv;
    Table t(DriverRef{&drv});
    std::set<uint64_t> live = {0x1111};
    const Connected conn{&live};
    const auto h = handleOf("same-bytes");
    const uint64_t old_engine = 0x1111, new_engine = 0x2222;
    auto old_map = t.acquire(old_engine, kProcA, 0, h.data(), conn);
    ASSERT_EQ(old_map.status, Table::Status::kOk);
    live = {new_engine};  // the old engine's connection closed
    auto c = t.acquire(new_engine, kProcA, 0, h.data(), conn);
    EXPECT_EQ(c.status, Table::Status::kConflict);
    EXPECT_EQ(c.other, old_engine);
    EXPECT_FALSE(c.other_connected);
    EXPECT_EQ(c.ptr, nullptr);
    EXPECT_EQ(t.conflicts(), 1u);
    EXPECT_EQ(drv.opens, 1);  // not opened: the driver would return the old
    // The old engine's last copy finishes.
    EXPECT_TRUE(t.release(old_engine, 0, h.data()));
    auto fresh = t.acquire(new_engine, kProcA, 0, h.data(), conn);
    ASSERT_EQ(fresh.status, Table::Status::kOk);
    EXPECT_TRUE(fresh.opened);
    EXPECT_EQ(t.mappingsOf(old_engine), 0u);
    EXPECT_EQ(t.mappingsOf(new_engine), 1u);
}

// Two processes that report the same pid and node (the first process of
// two containers of one pod) are told apart by their process tokens.
TEST(NvlinkProxyImports, SamePidAndNodeOfTwoProcessesIsNotShared) {
    FakeDriver drv;
    Table t(DriverRef{&drv});
    std::set<uint64_t> live = {0xA, 0xB, 0xC};
    const Connected conn{&live};
    const ImportOwner first{7, "node-1", 0x1234}, second{7, "node-1", 0x5678};
    const auto h = handleOf("same-bytes");
    ASSERT_EQ(t.acquire(0xA, first, 0, h.data(), conn).status,
              Table::Status::kOk);
    EXPECT_EQ(t.acquire(0xB, second, 0, h.data(), conn).status,
              Table::Status::kConflict);
    // Another engine of the first process still shares it.
    auto c = t.acquire(0xC, first, 0, h.data(), conn);
    ASSERT_EQ(c.status, Table::Status::kOk);
    EXPECT_TRUE(c.shared);
    EXPECT_EQ(drv.opens, 1);
}

TEST(NvlinkProxyImports, OpenErrorLeavesNothingBehind) {
    FakeDriver drv;
    drv.fail_with = 201;
    Table t(DriverRef{&drv});
    std::set<uint64_t> live = {0xA};
    const auto h = handleOf("pid-collision");
    auto a = t.acquire(0xA, kProcA, 0, h.data(), Connected{&live});
    EXPECT_EQ(a.status, Table::Status::kError);
    EXPECT_EQ(a.error, 201);
    EXPECT_EQ(t.mappings(), 0u);
    EXPECT_FALSE(t.release(0xA, 0, h.data()));
}

TEST(NvlinkProxyDaemonSetup, SocketModeParsing) {
    unsigned m = 0;
    EXPECT_TRUE(parseSocketMode("600", &m));
    EXPECT_EQ(m, 0600u);
    EXPECT_TRUE(parseSocketMode("0660", &m));
    EXPECT_EQ(m, 0660u);
    EXPECT_TRUE(parseSocketMode("0644", &m));
    EXPECT_EQ(m, 0644u);
    // Malformed, or without read and write for the owner (the daemon's own
    // --ping / --stats connect as the owner).
    for (const char *bad : {"", "0888", "1777", "77777", "rw", "-600", "0x1",
                            "0", "000", "0060", "0400", "0200", "0066"})
        EXPECT_FALSE(parseSocketMode(bad, &m)) << bad;

    EXPECT_EQ(kDefaultSocketMode, 0600u);
}

namespace {
std::string tempDir() {
    char tmpl[] = "/tmp/nvlp-sock-XXXXXX";
    const char *d = mkdtemp(tmpl);
    return d ? d : "";
}
}  // namespace

// The socket file has exactly the requested mode, whatever the umask, from
// the moment it exists.
TEST(NvlinkProxyDaemonSetup, SocketIsCreatedWithItsMode) {
    const std::string dir = tempDir();
    ASSERT_FALSE(dir.empty());
    const mode_t old = umask(0);  // a permissive umask must not widen it
    for (unsigned mode : {0600u, 0660u}) {
        const std::string path = dir + "/proxy.sock";
        std::string err;
        const int fd = listenUnixSocket(path, mode, 4, &err);
        ASSERT_GE(fd, 0) << err;
        struct stat st;
        ASSERT_EQ(lstat(path.c_str(), &st), 0);
        EXPECT_TRUE(S_ISSOCK(st.st_mode));
        EXPECT_EQ(st.st_mode & 0777, mode);
        close(fd);
        unlink(path.c_str());
    }
    umask(old);
    rmdir(dir.c_str());
}

// Owner-only: the daemon's user connects, another user is refused.
TEST(NvlinkProxyDaemonSetup, OwnerOnlySocketRefusesOtherUsers) {
    const std::string dir = tempDir();
    ASSERT_FALSE(dir.empty());
    chmod(dir.c_str(), 0755);
    const std::string path = dir + "/proxy.sock";
    std::string err;
    const int lfd = listenUnixSocket(path, kDefaultSocketMode, 4, &err);
    ASSERT_GE(lfd, 0) << err;
    auto tryConnect = [&]() {
        const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        memcpy(addr.sun_path, path.c_str(), path.size() + 1);
        const int rc =
            connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
        const int e = rc == 0 ? 0 : errno;
        close(fd);
        return e;
    };
    EXPECT_EQ(tryConnect(), 0);
    if (geteuid() != 0) {
        close(lfd);
        unlink(path.c_str());
        rmdir(dir.c_str());
        GTEST_SKIP() << "switching to another user needs root";
    }
    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        // A different, unprivileged user (no CAP_DAC_OVERRIDE).
        if (setgid(65534) != 0 || setuid(65534) != 0) _exit(3);
        _exit(tryConnect() == EACCES ? 0 : 1);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0)
        << "another user could connect to an owner-only socket";
    close(lfd);
    unlink(path.c_str());
    rmdir(dir.c_str());
}

TEST(NvlinkProxyDaemonSetup, PeerCredentials) {
    int sv[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    const PeerCred pc = peerCred(sv[0]);
    ASSERT_TRUE(pc.known);
    EXPECT_EQ(pc.uid, uint32_t(geteuid()));
    EXPECT_EQ(pc.pid, int64_t(getpid()));
    close(sv[0]);
    close(sv[1]);
}

TEST(NvlinkProxyDaemonSetup, PidCollisionAndVisibleGpus) {
    EXPECT_TRUE(pidValuesCollide(1, 1));
    EXPECT_FALSE(pidValuesCollide(42, 1));
    EXPECT_FALSE(pidValuesCollide(0, 0));  // unknown pid
    EXPECT_EQ(visibleGpuShortfall(1, 0), "");
    EXPECT_EQ(visibleGpuShortfall(2, 2), "");
    EXPECT_EQ(visibleGpuShortfall(8, 2), "");
    const std::string m = visibleGpuShortfall(1, 2);
    EXPECT_NE(m.find("only 1 GPU(s) visible"), std::string::npos) << m;
    EXPECT_NE(m.find("--min-visible-gpus 2"), std::string::npos) << m;
    EXPECT_NE(m.find("NVIDIA_VISIBLE_DEVICES=all"), std::string::npos) << m;
    EXPECT_NE(visibleGpuShortfall(0, 1), "");
}
