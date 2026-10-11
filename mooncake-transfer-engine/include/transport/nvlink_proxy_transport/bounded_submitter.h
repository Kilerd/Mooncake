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

#ifndef NVLINK_PROXY_BOUNDED_SUBMITTER_H_
#define NVLINK_PROXY_BOUNDED_SUBMITTER_H_

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

#include "transport/transport.h"

namespace mooncake {

// Runs requests on another transport with backpressure: at most
// |max_inflight| requests (shared by all callers) are submitted at a time,
// in chunks of at most |chunk| requests, and requests that fail (e.g. a full
// TCP lane queue) are retried with backoff until |deadline_ns|. Used by the
// nvlink_proxy transport to move same-node traffic onto its base transport
// without flooding it when the copy daemon is unavailable.
class BoundedSubmitter {
   public:
    struct Result {
        size_t succeeded = 0;
        size_t failed = 0;     // failed for good (deadline passed)
        size_t retried = 0;    // request attempts that were retried
        size_t abandoned = 0;  // still unresolved long after the deadline
    };

    BoundedSubmitter(size_t max_inflight, size_t chunk);

    // A private attempt batch that was not settled by |abandon_ns| (a task
    // still in flight, or a cut transfer its receiver has not confirmed).
    // The callee owns |batch| from now on: it frees it once settled.
    using OnAbandon = std::function<void(Transport::BatchDesc *batch)>;
    // An attempt that settled with transfers its receiver can never confirm
    // (BatchControl::unprovable): called before its requests are reported;
    // the caller's batches never quiesce.
    using OnUnprovable = std::function<void()>;

    // Blocks until every request succeeded or failed for good; calls
    // |on_done(index, ok)| exactly once per request. |deadline_ns| is in the
    // steady_clock domain (nanoseconds since its epoch) and is the deadline
    // of every attempt batch: a transport that honours batch deadlines (TCP)
    // stops the work there; work still running a second later is canceled.
    // An attempt is waited for until it is settled -- every task terminal
    // and no cut transfer left that its receiver may still confirm -- before
    // its failed requests are retried; failed requests of an attempt with an
    // unprovable transfer are not retried (on_unprovable). One that is not
    // settled when |abandon_ns| passes reports its unfinished requests as
    // failed and is handed to |on_abandon| (or, when that is empty, leaked
    // rather than freed under the transport). |canceled| (optional): the
    // caller's batch was canceled; the attempt in flight is canceled then
    // too.
    Result run(Transport *transport,
               const std::vector<const Transport::TransferRequest *> &requests,
               int64_t deadline_ns, int64_t abandon_ns,
               const std::function<void(size_t, bool)> &on_done,
               const OnAbandon &on_abandon = nullptr,
               const std::function<bool()> &canceled = nullptr,
               const OnUnprovable &on_unprovable = nullptr);

    // Every task of |batch| is terminal and its control has no unconfirmed
    // attempt left.
    static bool quiesced(Transport::BatchDesc &batch);
    // Every task of |batch| is terminal and every unconfirmed attempt left
    // is unprovable: no transport uses it, nothing more will be confirmed.
    static bool settled(Transport::BatchDesc &batch);

    size_t inflight() const;

   private:
    void acquire(size_t n);
    void release(size_t n);

    const size_t max_inflight_;
    const size_t chunk_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    size_t inflight_ = 0;
};

}  // namespace mooncake

#endif  // NVLINK_PROXY_BOUNDED_SUBMITTER_H_
