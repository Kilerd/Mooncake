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

// Waiting for a batch to be quiesced, and the retrying synchronous submit of
// the Python binding built on it.
//
// A batch is quiesced when every task is terminal and no attempt is left
// unconfirmed (Transport::BatchProgress): nothing of it, of any attempt, can
// still be applied anywhere. Only then may it be freed or its requests be
// submitted again; a batch that is not quiesced in time is neither freed
// nor retried (its writes may still land).
//
// Header-only and templated on the engine (TransferEngine in the binding, a
// fake in the unit test).

#ifndef MOONCAKE_BATCH_QUIESCE_H_
#define MOONCAKE_BATCH_QUIESCE_H_

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#include "transport/transport.h"

namespace mooncake {

inline int64_t batchQuiesceNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Polls |batch| until it is quiesced or |timeout_ms| passed (0: one look).
// True when quiesced; |last| receives the last progress seen.
template <class Engine>
bool waitBatchQuiesced(Engine &engine, Transport::BatchID batch,
                       int64_t timeout_ms,
                       Transport::BatchProgress *last = nullptr) {
    const int64_t until =
        batchQuiesceNowNs() + std::max<int64_t>(0, timeout_ms) * 1000000LL;
    int spins = 0;
    while (true) {
        Transport::BatchProgress p;
        if (!engine.getBatchProgress(batch, p).ok()) return false;
        if (last) *last = p;
        if (p.quiesced) return true;
        if (batchQuiesceNowNs() >= until) return false;
        if (++spins < 1000) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
}

enum class SyncOutcome {
    kCompleted,   // every request completed; the batch was freed
    kFailed,      // failed or timed out; quiesced and freed
    kUnquiesced,  // failed or timed out and not quiesced in time: writes
                  // may still land; the batch was neither freed nor retried
};

struct SyncOptions {
    int max_attempts = 1;
    int64_t start_ns = 0;             // steady_clock
    int64_t timeout_ns = 0;           // whole call, from start_ns; 0 = none
    int64_t quiesce_wait_ms = 10000;  // after a failure or timeout
};

// Submits one batch of |batch_size| requests per attempt (|submit| fills
// it), with the call's deadline as the batch deadline, and polls it:
// - all tasks completed and the batch quiesced: freed, kCompleted;
// - a task failed, or the deadline passed: the batch is canceled and waited
//   for until quiesced (at most quiesce_wait_ms). Not quiesced: kUnquiesced
//   (the batch is left allocated). Quiesced: freed; a failure is retried
//   while attempts and time remain, else kFailed.
// |submit(batch, attempt)| submits to an allocated batch; the requests it
// passes must outlive the batch (an unquiesced batch keeps referencing
// them). |on_submit_error| runs when it fails (e.g. drop a cached segment
// handle).
template <class Engine>
SyncOutcome runSyncWithRetries(
    Engine &engine, size_t batch_size, const SyncOptions &opt,
    const std::function<Status(Transport::BatchID, int)> &submit,
    const std::function<void()> &on_submit_error = nullptr) {
    const int64_t deadline =
        opt.timeout_ns > 0 ? opt.start_ns + opt.timeout_ns : 0;
    auto settle = [&](Transport::BatchID batch) -> bool {
        // Nothing more of this attempt may be applied before it is freed or
        // retried.
        engine.cancelBatch(batch);
        if (!waitBatchQuiesced(engine, batch, opt.quiesce_wait_ms))
            return false;  // left allocated: freeing it would be unsafe
        engine.freeBatchID(batch);
        return true;
    };
    for (int attempt = 0; attempt < std::max(1, opt.max_attempts); ++attempt) {
        auto batch = engine.allocateBatchID(batch_size);
        if (deadline) engine.setBatchDeadline(batch, deadline);
        Status s = submit(batch, attempt);
        if (!s.ok()) {
            if (on_submit_error) on_submit_error();
            return settle(batch) ? SyncOutcome::kFailed
                                 : SyncOutcome::kUnquiesced;
        }
        bool retry = false;
        int spins = 0;
        while (true) {
            Transport::BatchProgress p;
            if (!engine.getBatchProgress(batch, p).ok())
                return settle(batch) ? SyncOutcome::kFailed
                                     : SyncOutcome::kUnquiesced;
            if (p.all_terminal && p.failed == 0) {
                // Every request completed. Freed once quiesced (an attempt
                // of it, e.g. a GPU copy without an answer, may still be
                // unconfirmed).
                if (!p.quiesced &&
                    !waitBatchQuiesced(engine, batch, opt.quiesce_wait_ms))
                    return SyncOutcome::kUnquiesced;
                engine.freeBatchID(batch);
                return SyncOutcome::kCompleted;
            }
            const bool timed_out = deadline && batchQuiesceNowNs() > deadline;
            if (p.failed > 0 || timed_out) {
                if (!settle(batch)) return SyncOutcome::kUnquiesced;
                if (timed_out) return SyncOutcome::kFailed;
                retry = true;
                break;
            }
            if (++spins < 1000) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
        if (!retry) break;
        if (deadline && batchQuiesceNowNs() > deadline) break;
    }
    return SyncOutcome::kFailed;
}

}  // namespace mooncake

#endif  // MOONCAKE_BATCH_QUIESCE_H_
