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

    // Blocks until every request succeeded or failed for good; calls
    // |on_done(index, ok)| exactly once per request. |deadline_ns| is in the
    // steady_clock domain (nanoseconds since its epoch). A request still in
    // flight on |transport| when |abandon_ns| passes is reported as failed and
    // its bookkeeping is leaked rather than freed under the transport.
    Result run(Transport *transport,
               const std::vector<const Transport::TransferRequest *> &requests,
               int64_t deadline_ns, int64_t abandon_ns,
               const std::function<void(size_t, bool)> &on_done);

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
