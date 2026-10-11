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

// The retrying synchronous submit of the Python binding (batch_quiesce.h)
// against a scripted engine: an attempt is retried or freed only once it is
// quiesced, and one that never quiesces is neither.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "batch_quiesce.h"

using namespace mooncake;

namespace {

// One attempt's script: when (ms after submit) its tasks become terminal,
// whether they fail, and when (ms after cancel; -1 = never) a cut attempt
// is confirmed.
struct Script {
    int terminal_after_ms = 0;
    bool fail = false;
    int64_t unconfirmed = 0;          // left after the tasks end
    int confirm_after_cancel_ms = 0;  // -1: never
    bool stop_on_cancel = true;       // cancel ends pending tasks at once
    bool stop_at_deadline = true;     // the batch deadline ends them too
};

class FakeEngine {
   public:
    explicit FakeEngine(std::vector<Script> scripts)
        : scripts_(std::move(scripts)) {}

    Transport::BatchID allocateBatchID(size_t) {
        std::lock_guard<std::mutex> lock(mu_);
        Batch b;
        b.attempt = allocated_++;
        const auto id = next_id_++;
        batches_[id] = b;
        return id;
    }
    Status freeBatchID(Transport::BatchID id) {
        std::lock_guard<std::mutex> lock(mu_);
        Transport::BatchProgress p = progressLocked(id);
        if (!p.quiesced) {
            freed_unquiesced_ = true;
            return Status::BatchBusy("not quiesced");
        }
        batches_.erase(id);
        ++freed_;
        return Status::OK();
    }
    Status setBatchDeadline(Transport::BatchID id, int64_t deadline) {
        std::lock_guard<std::mutex> lock(mu_);
        batches_[id].deadline = deadline;
        return Status::OK();
    }
    Status cancelBatch(Transport::BatchID id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto &b = batches_[id];
        if (!b.canceled) {
            b.canceled = true;
            b.canceled_at = now();
        }
        return Status::OK();
    }
    Status getBatchProgress(Transport::BatchID id,
                            Transport::BatchProgress &p) {
        std::lock_guard<std::mutex> lock(mu_);
        p = progressLocked(id);
        return Status::OK();
    }
    Status submit(Transport::BatchID id) {
        std::lock_guard<std::mutex> lock(mu_);
        // Never while an earlier attempt can still write.
        for (auto &kv : batches_) {
            if (kv.first == id) continue;
            if (!progressLocked(kv.first).quiesced)
                submitted_while_unquiesced_ = true;
        }
        auto &b = batches_[id];
        b.submitted_at = now();
        ++submits_;
        return Status::OK();
    }

    int submits() const { return submits_; }
    int freed() const { return freed_; }
    bool submittedWhileUnquiesced() const {
        return submitted_while_unquiesced_;
    }
    bool freedUnquiesced() const { return freed_unquiesced_; }
    size_t live() const { return batches_.size(); }

   private:
    struct Batch {
        int attempt = 0;
        int64_t submitted_at = 0, deadline = 0, canceled_at = 0;
        bool canceled = false;
    };
    static int64_t now() { return batchQuiesceNowNs(); }

    Transport::BatchProgress progressLocked(Transport::BatchID id) {
        Transport::BatchProgress p;
        auto &b = batches_[id];
        const Script &s =
            scripts_[std::min<size_t>(b.attempt, scripts_.size() - 1)];
        const int64_t t = now();
        p.total = 1;
        bool terminal =
            b.submitted_at &&
            t - b.submitted_at >= int64_t(s.terminal_after_ms) * 1000000;
        if (s.stop_at_deadline && b.deadline && t >= b.deadline)
            terminal = true;
        if (b.canceled && s.stop_on_cancel) terminal = true;
        const bool failed =
            s.fail || (terminal && !(t - b.submitted_at >=
                                     int64_t(s.terminal_after_ms) * 1000000));
        if (!terminal) {
            p.pending = 1;
        } else if (failed) {
            p.failed = 1;
        } else {
            p.completed = 1;
        }
        p.all_terminal = terminal;
        p.unconfirmed = 0;
        if (terminal && s.unconfirmed) {
            const bool confirmed =
                b.canceled && s.confirm_after_cancel_ms >= 0 &&
                t - b.canceled_at >=
                    int64_t(s.confirm_after_cancel_ms) * 1000000;
            if (!confirmed) p.unconfirmed = s.unconfirmed;
        }
        p.quiesced = p.all_terminal && p.unconfirmed == 0;
        return p;
    }

    std::vector<Script> scripts_;
    std::mutex mu_;
    std::map<Transport::BatchID, Batch> batches_;
    Transport::BatchID next_id_ = 0x1000;
    int allocated_ = 0, submits_ = 0, freed_ = 0;
    bool submitted_while_unquiesced_ = false, freed_unquiesced_ = false;
};

SyncOutcome runWith(FakeEngine &engine, int attempts, int64_t timeout_ms,
                    int64_t quiesce_ms) {
    SyncOptions opt;
    opt.max_attempts = attempts;
    opt.start_ns = batchQuiesceNowNs();
    opt.timeout_ns = timeout_ms * 1000000;
    opt.quiesce_wait_ms = quiesce_ms;
    return runSyncWithRetries(engine, 1, opt, [&](Transport::BatchID b, int) {
        return engine.submit(b);
    });
}

}  // namespace

TEST(BatchQuiesce, CompletedAttemptIsFreed) {
    FakeEngine e({Script{5, false}});
    EXPECT_EQ(runWith(e, 3, 5000, 1000), SyncOutcome::kCompleted);
    EXPECT_EQ(e.submits(), 1);
    EXPECT_EQ(e.freed(), 1);
    EXPECT_EQ(e.live(), 0u);
}

// A failed attempt with a cut transfer still unconfirmed: canceled, waited
// for until the receiver confirmed, then retried -- never before.
TEST(BatchQuiesce, RetryWaitsUntilTheFailedAttemptIsQuiesced) {
    FakeEngine e({Script{5, true, 1, 50}, Script{5, false}});
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(runWith(e, 3, 5000, 1000), SyncOutcome::kCompleted);
    EXPECT_GE(std::chrono::steady_clock::now() - t0,
              std::chrono::milliseconds(50));
    EXPECT_EQ(e.submits(), 2);
    EXPECT_FALSE(e.submittedWhileUnquiesced());
    EXPECT_FALSE(e.freedUnquiesced());
    EXPECT_EQ(e.live(), 0u);
}

// The failed attempt never quiesces: no retry, not freed, kUnquiesced.
TEST(BatchQuiesce, UnquiescedAttemptIsNeitherRetriedNorFreed) {
    FakeEngine e({Script{5, true, 1, -1}, Script{5, false}});
    EXPECT_EQ(runWith(e, 3, 5000, 100), SyncOutcome::kUnquiesced);
    EXPECT_EQ(e.submits(), 1);
    EXPECT_EQ(e.freed(), 0);
    EXPECT_EQ(e.live(), 1u);
    EXPECT_FALSE(e.submittedWhileUnquiesced());
}

// The call's deadline passes with the task still running: canceled, waited
// for, freed, kFailed (no retry after a timeout).
TEST(BatchQuiesce, TimeoutCancelsAndWaitsBeforeFreeing) {
    FakeEngine e({Script{10000, false, 1, 30, true}});
    EXPECT_EQ(runWith(e, 3, 50, 1000), SyncOutcome::kFailed);
    EXPECT_EQ(e.submits(), 1);
    EXPECT_EQ(e.freed(), 1);
    EXPECT_FALSE(e.freedUnquiesced());
}

// A pending task that ignores cancel: the attempt is not quiesced in time.
TEST(BatchQuiesce, PendingTaskThatIgnoresCancelIsReportedUnquiesced) {
    FakeEngine e({Script{100000, false, 0, 0, false, false}});
    EXPECT_EQ(runWith(e, 3, 50, 100), SyncOutcome::kUnquiesced);
    EXPECT_EQ(e.freed(), 0);
    EXPECT_EQ(e.live(), 1u);
}

TEST(BatchQuiesce, WaitReportsFalseWhileWritesMayStillLand) {
    FakeEngine e({Script{80, true, 1, 40}});
    auto b = e.allocateBatchID(1);
    ASSERT_TRUE(e.submit(b).ok());
    Transport::BatchProgress p;
    EXPECT_FALSE(waitBatchQuiesced(e, b, 20, &p));  // still pending
    EXPECT_EQ(p.pending, 1u);
    EXPECT_FALSE(waitBatchQuiesced(e, b, 150, &p));  // failed, unconfirmed
    EXPECT_TRUE(p.all_terminal);
    EXPECT_EQ(p.unconfirmed, 1);
    ASSERT_TRUE(e.cancelBatch(b).ok());
    EXPECT_TRUE(waitBatchQuiesced(e, b, 1000, &p));
    EXPECT_EQ(p.failed, 1u);
    EXPECT_TRUE(e.freeBatchID(b).ok());
}
