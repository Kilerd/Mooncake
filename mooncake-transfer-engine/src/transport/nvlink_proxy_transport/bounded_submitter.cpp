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

#include "transport/nvlink_proxy_transport/bounded_submitter.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <thread>

namespace mooncake {

namespace {
int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

BoundedSubmitter::BoundedSubmitter(size_t max_inflight, size_t chunk)
    : max_inflight_(std::max<size_t>(1, max_inflight)),
      chunk_(std::max<size_t>(1, std::min(chunk, max_inflight_))) {}

size_t BoundedSubmitter::inflight() const {
    std::lock_guard<std::mutex> lock(mu_);
    return inflight_;
}

void BoundedSubmitter::acquire(size_t n) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return inflight_ + n <= max_inflight_; });
    inflight_ += n;
}

void BoundedSubmitter::release(size_t n) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        inflight_ -= n;
    }
    cv_.notify_all();
}

BoundedSubmitter::Result BoundedSubmitter::run(
    Transport *transport,
    const std::vector<const Transport::TransferRequest *> &requests,
    int64_t deadline_ns, int64_t abandon_ns,
    const std::function<void(size_t, bool)> &on_done) {
    using TransferStatusEnum = Transport::TransferStatusEnum;
    Result result;
    std::vector<size_t> pending(requests.size());
    for (size_t i = 0; i < pending.size(); ++i) pending[i] = i;
    int round = 0;
    while (!pending.empty()) {
        std::vector<size_t> failed;
        for (size_t begin = 0; begin < pending.size(); begin += chunk_) {
            const size_t n = std::min(chunk_, pending.size() - begin);
            acquire(n);
            // Private batch: the caller's tasks keep their own status; these
            // tasks only carry this attempt.
            auto *batch = new Transport::BatchDesc();
            batch->id = reinterpret_cast<Transport::BatchID>(batch);
            batch->batch_size = n;
            batch->context = nullptr;
            batch->task_list.resize(n);
            std::vector<Transport::TransferTask *> tasks(n);
            for (size_t k = 0; k < n; ++k) {
                auto &task = batch->task_list[k];
                task.batch_id = batch->id;
                task.request = requests[pending[begin + k]];
                task.transport_ = transport;
                tasks[k] = &task;
            }
            Status submitted = transport->submitTransferTask(tasks);
            std::vector<int> state(n, 0);  // 0 running, 1 ok, 2 failed
            size_t open = n;
            if (!submitted.ok()) {
                // Tasks the transport never started cannot complete.
                for (size_t k = 0; k < n; ++k) {
                    if (batch->task_list[k].slice_count == 0) {
                        state[k] = 2;
                        --open;
                    }
                }
            }
            int spins = 0;
            bool abandoned = false;
            while (open > 0) {
                for (size_t k = 0; k < n; ++k) {
                    if (state[k]) continue;
                    Transport::TransferStatus st;
                    Status s = transport->getTransferStatus(batch->id, k, st);
                    if (!s.ok() || st.s == TransferStatusEnum::FAILED ||
                        st.s == TransferStatusEnum::TIMEOUT ||
                        st.s == TransferStatusEnum::CANCELED) {
                        state[k] = 2;
                        --open;
                    } else if (st.s == TransferStatusEnum::COMPLETED) {
                        state[k] = 1;
                        --open;
                    }
                }
                if (open == 0) break;
                if (nowNs() > abandon_ns) {
                    abandoned = true;
                    break;
                }
                if (++spins < 64) {
                    std::this_thread::yield();
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(
                        std::min(1000, 20 * (spins - 63))));
                }
            }
            release(n);
            for (size_t k = 0; k < n; ++k) {
                const size_t idx = pending[begin + k];
                if (state[k] == 1) {
                    ++result.succeeded;
                    on_done(idx, true);
                } else if (state[k] == 2) {
                    failed.push_back(idx);
                } else {
                    ++result.abandoned;
                    ++result.failed;
                    on_done(idx, false);
                }
            }
            if (abandoned) {
                // The transport may still touch these tasks; leak them.
                LOG(ERROR) << "nvlink_proxy: " << result.abandoned
                           << " fallback request(s) still unresolved long "
                              "after their deadline; reporting them failed";
            } else {
                delete batch;
            }
        }
        if (failed.empty()) break;
        const int64_t now = nowNs();
        if (now >= deadline_ns) {
            for (size_t idx : failed) {
                ++result.failed;
                on_done(idx, false);
            }
            break;
        }
        result.retried += failed.size();
        // Back off before retrying so a full queue can drain.
        const int64_t backoff_ns =
            std::min<int64_t>(200, 5LL << std::min(round, 6)) * 1000000LL;
        std::this_thread::sleep_for(std::chrono::nanoseconds(
            std::min<int64_t>(backoff_ns, deadline_ns - now)));
        ++round;
        pending.swap(failed);
    }
    return result;
}

}  // namespace mooncake
