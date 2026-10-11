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

// Two lanes ("GPUs"), each chunk writes its index into its own slot of
// |dst| after |chunk_ms|; records when every write landed.
struct PacedRig {
    std::map<int, std::unique_ptr<FakeStream>> streams;
    std::map<int, std::vector<std::unique_ptr<FakeEvent>>> events;
    std::vector<std::atomic<int>> dst;
    std::vector<uint64_t> landed_us;
    std::mutex landed_mu;
    std::atomic<size_t> max_outstanding{0};
    std::map<int, std::atomic<size_t>> outstanding;
    int chunk_ms;

    PacedRig(size_t chunks, int chunk_ms_) : dst(chunks), chunk_ms(chunk_ms_) {
        for (auto &d : dst) d = -1;
        for (int lane : {0, 1}) {
            streams[lane].reset(new FakeStream());
            for (size_t i = 0; i < kPacedWindow; ++i)
                events[lane].emplace_back(new FakeEvent());
            outstanding[lane] = 0;
        }
    }

    PacedResult run(const std::vector<int> &lanes, uint64_t deadline_us,
                    int fail_at = -1) {
        return issuePaced(
            lanes, kPacedWindow, deadline_us, nowUs,
            [&](size_t k) -> int {
                if (int(k) == fail_at) return 700;
                const int lane = lanes[k];
                const size_t now_out = ++outstanding[lane];
                size_t seen = max_outstanding.load();
                while (now_out > seen &&
                       !max_outstanding.compare_exchange_weak(seen, now_out)) {
                }
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
                return 0;
            },
            [&](int lane, size_t slot) -> int {
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
            [&]() -> int {
                for (auto &kv : streams) kv.second->sync();
                return 0;
            });
    }

    std::vector<int> snapshot() {
        std::vector<int> out;
        for (auto &d : dst) out.push_back(d.load());
        return out;
    }
};

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

// No deadline: every chunk is issued and has landed when issuePaced
// returns; never more than the window per lane in flight.
TEST(NvlinkProxyPacedIssue, IssuesEverythingWithinTheWindow) {
    constexpr size_t kChunks = 12;
    PacedRig rig(kChunks, 5);
    std::vector<int> lanes;
    for (size_t k = 0; k < kChunks; ++k) lanes.push_back(int(k % 2));
    const PacedResult r = rig.run(lanes, UINT64_MAX);
    EXPECT_EQ(r.issued, kChunks);
    EXPECT_FALSE(r.deadline);
    EXPECT_EQ(r.error, 0);
    for (size_t k = 0; k < kChunks; ++k) EXPECT_EQ(rig.dst[k].load(), int(k));
    EXPECT_LE(rig.max_outstanding.load(), kPacedWindow);
}

// The deadline passes mid-copy: nothing more is issued, what is in flight
// lands before issuePaced returns (the reply), and nothing lands after it.
TEST(NvlinkProxyPacedIssue, NoWriteLandsAfterTheDeadlineRefusal) {
    constexpr size_t kChunks = 40;
    constexpr int kChunkMs = 10;
    PacedRig rig(kChunks, kChunkMs);
    const std::vector<int> lanes(kChunks, 0);  // one GPU: 400 ms of work
    const uint64_t start = nowUs();
    const uint64_t deadline = start + 100 * 1000;
    const PacedResult r = rig.run(lanes, deadline);
    const uint64_t reply = nowUs();
    EXPECT_TRUE(r.deadline);
    EXPECT_EQ(r.error, 0);
    EXPECT_LT(r.issued, kChunks);
    EXPECT_GE(r.issued, 5u);
    // Drained: the reply leaves at most the window's chunks after the
    // deadline (plus scheduling slack).
    EXPECT_LE(reply - deadline, uint64_t(kPacedWindow * kChunkMs + 40) * 1000);
    EXPECT_EQ(rig.streams[0]->queued(), 0u);
    const std::vector<int> at_reply = rig.snapshot();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    EXPECT_EQ(rig.snapshot(), at_reply) << "a write landed after the reply";
    {
        std::lock_guard<std::mutex> l(rig.landed_mu);
        ASSERT_EQ(rig.landed_us.size(), r.issued);
        for (uint64_t t : rig.landed_us) EXPECT_LE(t, reply);
    }
    // Exactly the issued prefix was written.
    for (size_t k = 0; k < kChunks; ++k)
        EXPECT_EQ(at_reply[k], k < r.issued ? int(k) : -1) << k;
}

TEST(NvlinkProxyPacedIssue, ErrorStopsIssueAndStillDrains) {
    constexpr size_t kChunks = 10;
    PacedRig rig(kChunks, 5);
    std::vector<int> lanes;
    for (size_t k = 0; k < kChunks; ++k) lanes.push_back(int(k % 2));
    const PacedResult r = rig.run(lanes, UINT64_MAX, 6);
    EXPECT_EQ(r.error, 700);
    EXPECT_EQ(r.issued, 6u);
    EXPECT_EQ(rig.streams[0]->queued() + rig.streams[1]->queued(), 0u);
    for (size_t k = 0; k < kChunks; ++k)
        EXPECT_EQ(rig.dst[k].load(), k < 6 ? int(k) : -1);
}

TEST(NvlinkProxyPacedIssue, DeadlineAlreadyPassedIssuesNothing) {
    PacedRig rig(4, 1);
    const PacedResult r = rig.run({0, 1, 0, 1}, 0);
    EXPECT_TRUE(r.deadline);
    EXPECT_EQ(r.issued, 0u);
}

namespace {

// Behaves like the driver: one mapping per (context, handle); opening it
// again where it is open returns the same mapping; one close unmaps it for
// every user (|live| says which mappings still exist).
struct FakeDriver {
    std::map<std::pair<int, std::string>, uintptr_t> open_maps;
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
            it = open_maps.emplace(key, next).first;
            live[next] = true;
            next += 0x1000;
        }
        ++opens;
        *ptr = reinterpret_cast<void *>(it->second);
        return 0;
    }
    int close(int, void *ptr) {
        const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        if (!live[p]) return 400;
        live[p] = false;
        for (auto it = open_maps.begin(); it != open_maps.end();)
            it = it->second == p ? open_maps.erase(it) : std::next(it);
        ++closes;
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

}  // namespace

// An engine disconnects: once its registrations are gone, every mapping it
// had (in every GPU context) is closed and nothing of it stays open.
TEST(NvlinkProxyImports, DisconnectClosesEveryMappingOfTheEngine) {
    FakeDriver drv;
    ImportTable<DriverRef> t(DriverRef{&drv});
    const uint64_t engine = 0xA, other = 0xB;
    const auto h1 = handleOf("kv-pool-1"), h2 = handleOf("kv-pool-2"),
               hb = handleOf("other-engine");
    // Two blocks, each mapped in its own GPU's context and the peer's.
    for (int dev : {0, 1}) {
        ASSERT_EQ(t.acquire(engine, dev, h1.data()).status,
                  ImportTable<DriverRef>::Status::kOk);
        ASSERT_EQ(t.acquire(engine, dev, h2.data()).status,
                  ImportTable<DriverRef>::Status::kOk);
    }
    ASSERT_EQ(t.acquire(other, 1, hb.data()).status,
              ImportTable<DriverRef>::Status::kOk);
    EXPECT_EQ(t.mappingsOf(engine), 4u);
    EXPECT_EQ(t.mappings(), 5u);
    for (int dev : {0, 1}) {
        EXPECT_TRUE(t.release(engine, dev, h1.data()));
        EXPECT_TRUE(t.release(engine, dev, h2.data()));
    }
    EXPECT_EQ(t.mappingsOf(engine), 0u);
    EXPECT_EQ(t.mappings(), 1u);  // the other engine's stays
    EXPECT_EQ(drv.closes, 4);
    EXPECT_EQ(t.opens(), 5u);
    EXPECT_EQ(t.closes(), 4u);
}

// The same engine registers a block again (reconnect) while a copy still
// holds the old registration: both share one mapping, which stays usable
// until the last of them is gone (the driver would unmap it for both).
TEST(NvlinkProxyImports, ReRegistrationSharesTheMapping) {
    FakeDriver drv;
    ImportTable<DriverRef> t(DriverRef{&drv});
    const auto h = handleOf("kv-pool");
    auto a = t.acquire(0xA, 0, h.data());
    auto b = t.acquire(0xA, 0, h.data());
    ASSERT_EQ(b.status, ImportTable<DriverRef>::Status::kOk);
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

// A restarted engine (new client id) whose handle bytes repeat its
// predecessor's while a copy of the predecessor still holds that mapping:
// never served from the old mapping; served once it is closed.
TEST(NvlinkProxyImports, RestartedEngineIsNeverServedFromAStaleMapping) {
    FakeDriver drv;
    ImportTable<DriverRef> t(DriverRef{&drv});
    const auto h = handleOf("same-bytes");
    const uint64_t old_engine = 0x1111, new_engine = 0x2222;
    auto old_map = t.acquire(old_engine, 0, h.data());
    ASSERT_EQ(old_map.status, ImportTable<DriverRef>::Status::kOk);
    auto c = t.acquire(new_engine, 0, h.data());
    EXPECT_EQ(c.status, ImportTable<DriverRef>::Status::kConflict);
    EXPECT_EQ(c.other, old_engine);
    EXPECT_EQ(c.ptr, nullptr);
    EXPECT_EQ(t.conflicts(), 1u);
    EXPECT_EQ(drv.opens, 1);  // not even opened (the driver would dedupe)
    // The old engine's last copy finishes.
    EXPECT_TRUE(t.release(old_engine, 0, h.data()));
    auto fresh = t.acquire(new_engine, 0, h.data());
    ASSERT_EQ(fresh.status, ImportTable<DriverRef>::Status::kOk);
    EXPECT_TRUE(fresh.opened);
    EXPECT_EQ(t.mappingsOf(old_engine), 0u);
    EXPECT_EQ(t.mappingsOf(new_engine), 1u);
    // Another context is a separate mapping, owned by whoever opened it.
    EXPECT_EQ(t.acquire(old_engine, 1, h.data()).status,
              ImportTable<DriverRef>::Status::kOk);
}

TEST(NvlinkProxyImports, OpenErrorLeavesNothingBehind) {
    FakeDriver drv;
    drv.fail_with = 201;
    ImportTable<DriverRef> t(DriverRef{&drv});
    const auto h = handleOf("pid-collision");
    auto a = t.acquire(0xA, 0, h.data());
    EXPECT_EQ(a.status, ImportTable<DriverRef>::Status::kError);
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
    EXPECT_TRUE(parseSocketMode("0", &m));
    EXPECT_EQ(m, 0u);
    for (const char *bad : {"", "0888", "1777", "77777", "rw", "-600", "0x1"})
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
