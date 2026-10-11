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

namespace {
bool allTasksTerminal(Transport::BatchDesc &batch) {
    for (auto &task : batch.task_list) {
        const uint64_t slices =
            __atomic_load_n(&task.slice_count, __ATOMIC_ACQUIRE);
        const uint64_t done =
            __atomic_load_n(&task.success_slice_count, __ATOMIC_ACQUIRE) +
            __atomic_load_n(&task.failed_slice_count, __ATOMIC_ACQUIRE);
        if (slices != 0 && done != slices) return false;
    }
    return true;
}
}  // namespace

bool BoundedSubmitter::quiesced(Transport::BatchDesc &batch) {
    return allTasksTerminal(batch) &&
           batch.control.unconfirmed()->load(std::memory_order_acquire) <= 0;
}

bool BoundedSubmitter::settled(Transport::BatchDesc &batch) {
    if (!allTasksTerminal(batch)) return false;
    // Unprovable first (it is counted in unconfirmed first).
    const int64_t unprovable =
        batch.control.unprovable()->load(std::memory_order_acquire);
    return batch.control.unconfirmed()->load(std::memory_order_acquire) <=
           unprovable;
}

BoundedSubmitter::Result BoundedSubmitter::run(
    Transport *transport,
    const std::vector<const Transport::TransferRequest *> &requests,
    int64_t deadline_ns, int64_t abandon_ns,
    const std::function<void(size_t, bool)> &on_done,
    const OnAbandon &on_abandon, const std::function<bool()> &canceled_by,
    const OnUnprovable &on_unprovable) {
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
            // The attempt ends at the deadline (the transport stops it).
            batch->control.setDeadlineNs(deadline_ns);
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
            bool canceled = false;
            // Until every task is terminal, and then until the attempt is
            // settled (no cut transfer its receiver may still confirm): a
            // failed request is not retried while an earlier attempt of it
            // may still be applied.
            while (open > 0 || !settled(*batch)) {
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
                if (open == 0 && settled(*batch)) break;
                const int64_t now = nowNs();
                if (!canceled && open > 0 &&
                    (now > deadline_ns + 1000LL * 1000 * 1000 ||
                     (canceled_by && canceled_by()))) {
                    // The caller canceled, or a transport that did not stop
                    // at the deadline itself.
                    canceled = true;
                    batch->control.cancel();
                    transport->onBatchCanceled(batch->id);
                }
                if (now > abandon_ns) {
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
            if (abandoned) {
                // The transport may still touch these tasks. Handed over
                // first: the caller counts the attempt unconfirmed before
                // its requests are reported failed below.
                if (on_abandon) on_abandon(batch);  // else leaked
            }
            // Settled, but a transfer of it can never be confirmed: its
            // failed requests are not retried (it may still be applied),
            // and the caller learns of it before they are reported.
            const bool unprovable =
                !abandoned && batch->control.unprovable()->load(
                                  std::memory_order_acquire) > 0;
            if (unprovable && on_unprovable) on_unprovable();
            size_t abandoned_now = 0;
            for (size_t k = 0; k < n; ++k) {
                const size_t idx = pending[begin + k];
                if (state[k] == 1) {
                    ++result.succeeded;
                    on_done(idx, true);
                } else if (state[k] == 2 && !abandoned && !unprovable) {
                    failed.push_back(idx);
                } else if (state[k] == 2 && !abandoned) {
                    ++result.failed;  // not retried (see above)
                    on_done(idx, false);
                } else {
                    // Still running, or failed in an attempt that is not
                    // quiesced: never retried, failed for good.
                    ++abandoned_now;
                    ++result.abandoned;
                    ++result.failed;
                    on_done(idx, false);
                }
            }
            if (abandoned) {
                LOG(ERROR) << "nvlink_proxy: " << abandoned_now
                           << " fallback request(s) still unresolved long "
                              "after their deadline; reporting them failed "
                              "(their batch stays unquiesced until they end)";
            } else {
                delete batch;
            }
        }
        if (failed.empty()) break;
        const int64_t now = nowNs();
        if (now >= deadline_ns || (canceled_by && canceled_by())) {
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
