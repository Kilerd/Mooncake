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
// still be applied anywhere. Only then may its requests be submitted again;
// a failed batch that is not quiesced in time is not retried (its writes may
// still land). A batch is settled when it is quiesced, or when every task is
// terminal and the only unconfirmed attempts are unprovable ones (a receiver
// older than TCP protocol v3): no transport uses it any more and nothing
// more will ever be learned about it, so it may be freed -- it just never
// quiesces.
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

// See above. Reads unprovable before unconfirmed (BatchProgress), so
// unconfirmed - unprovable never undercounts the attempts still to confirm.
inline bool batchSettled(const Transport::BatchProgress &p) {
    return p.quiesced || (p.all_terminal && p.unconfirmed <= p.unprovable);
}

// Polls |batch| until |done(progress)| or |timeout_ms| passed (0: one
// look), or until |hopeless(progress)|. True when done; |last| receives the
// last progress seen.
template <class Engine, class Done, class Hopeless>
bool waitBatchUntil(Engine &engine, Transport::BatchID batch,
                    int64_t timeout_ms, Transport::BatchProgress *last,
                    Done done, Hopeless hopeless) {
    const int64_t until =
        batchQuiesceNowNs() + std::max<int64_t>(0, timeout_ms) * 1000000LL;
    int spins = 0;
    while (true) {
        Transport::BatchProgress p;
        if (!engine.getBatchProgress(batch, p).ok()) return false;
        if (last) *last = p;
        if (done(p)) return true;
        if (hopeless(p) || batchQuiesceNowNs() >= until) return false;
        if (++spins < 1000) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
}

// Polls |batch| until it is quiesced or |timeout_ms| passed (0: one look).
// True when quiesced; false at once when it never will be (an unprovable
// attempt). |last| receives the last progress seen.
template <class Engine>
bool waitBatchQuiesced(Engine &engine, Transport::BatchID batch,
                       int64_t timeout_ms,
                       Transport::BatchProgress *last = nullptr) {
    return waitBatchUntil(
        engine, batch, timeout_ms, last,
        [](const Transport::BatchProgress &p) { return p.quiesced; },
        [](const Transport::BatchProgress &p) {
            return p.unprovable > 0 && p.all_terminal;
        });
}

// Polls |batch| until it is settled or |timeout_ms| passed.
template <class Engine>
bool waitBatchSettled(Engine &engine, Transport::BatchID batch,
                      int64_t timeout_ms,
                      Transport::BatchProgress *last = nullptr) {
    return waitBatchUntil(
        engine, batch, timeout_ms, last,
        [](const Transport::BatchProgress &p) { return batchSettled(p); },
        [](const Transport::BatchProgress &) { return false; });
}

enum class SyncOutcome {
    kCompleted,   // every request completed (to a protocol v1 TCP receiver:
                  // sent); the batch was freed
    kFailed,      // failed or timed out; quiesced and freed
    kUnquiesced,  // failed or timed out and not quiesced in time, or never
                  // quiescable: writes may still land; the batch was not
                  // retried and is left allocated (*left_batch)
};

struct SyncOptions {
    int max_attempts = 1;
    int64_t start_ns = 0;             // steady_clock
    int64_t timeout_ns = 0;           // whole call, from start_ns; 0 = none
    int64_t quiesce_wait_ms = 10000;  // after a failure or timeout
};

// Submits one batch of |batch_size| requests per attempt (|submit| fills
// it), with the call's deadline as the batch deadline, and polls it:
// - all tasks completed: freed once settled (an attempt of it, e.g. a GPU
//   copy without an answer, may still be unconfirmed; waited for at most
//   quiesce_wait_ms), kCompleted;
// - a task failed, or the deadline passed: the batch is canceled and waited
//   for until quiesced (at most quiesce_wait_ms; not at all once it never
//   can be). Not quiesced: kUnquiesced, the batch is left allocated and
//   returned in |*left_batch| (the caller frees it once settled). Quiesced:
//   freed; a failure is retried while attempts and time remain, else
//   kFailed.
// |submit(batch, attempt)| submits to an allocated batch; the requests it
// passes must outlive the batch. |on_submit_error| runs when it fails
// (e.g. drop a cached segment handle).
template <class Engine>
SyncOutcome runSyncWithRetries(
    Engine &engine, size_t batch_size, const SyncOptions &opt,
    const std::function<Status(Transport::BatchID, int)> &submit,
    const std::function<void()> &on_submit_error = nullptr,
    Transport::BatchID *left_batch = nullptr) {
    const int64_t deadline =
        opt.timeout_ns > 0 ? opt.start_ns + opt.timeout_ns : 0;
    auto leave = [&](Transport::BatchID batch) {
        if (left_batch) *left_batch = batch;
        return SyncOutcome::kUnquiesced;
    };
    auto settle = [&](Transport::BatchID batch) -> bool {
        // Nothing more of this attempt may be applied before it is freed or
        // retried.
        engine.cancelBatch(batch);
        if (!waitBatchQuiesced(engine, batch, opt.quiesce_wait_ms))
            return false;  // left allocated
        engine.freeBatchID(batch);
        return true;
    };
    for (int attempt = 0; attempt < std::max(1, opt.max_attempts); ++attempt) {
        auto batch = engine.allocateBatchID(batch_size);
        if (deadline) engine.setBatchDeadline(batch, deadline);
        Status s = submit(batch, attempt);
        if (!s.ok()) {
            if (on_submit_error) on_submit_error();
            return settle(batch) ? SyncOutcome::kFailed : leave(batch);
        }
        bool retry = false;
        int spins = 0;
        while (true) {
            Transport::BatchProgress p;
            if (!engine.getBatchProgress(batch, p).ok())
                return settle(batch) ? SyncOutcome::kFailed : leave(batch);
            if (p.all_terminal && p.failed == 0) {
                if (!batchSettled(p) &&
                    !waitBatchSettled(engine, batch, opt.quiesce_wait_ms))
                    return leave(batch);
                engine.freeBatchID(batch);
                return SyncOutcome::kCompleted;
            }
            const bool timed_out = deadline && batchQuiesceNowNs() > deadline;
            if (p.failed > 0 || timed_out) {
                if (!settle(batch)) return leave(batch);
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
