// Copyright 2024 KVCache.AI
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

#ifndef TRANSPORT_H_
#define TRANSPORT_H_

#include <bits/stdint-uintn.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <atomic>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <unordered_map>

#include "common/base/status.h"
#include "transfer_metadata.h"

namespace mooncake {

class RdmaEndPoint;
class TransferMetadata;
/// By default, these functions return 0 (or non-null pointer) on success and
/// return -1 (or null pointer) on failure. The errno is set accordingly on
/// failure.
class Transport {
    friend class TransferEngine;
    friend class TransferEngineImpl;
    friend class MultiTransport;

   public:
    using SegmentID = uint64_t;
    using SegmentHandle = SegmentID;

    using BatchID = uint64_t;

    using BufferDesc = TransferMetadata::BufferDesc;
    using SegmentDesc = TransferMetadata::SegmentDesc;
    using HandShakeDesc = TransferMetadata::HandShakeDesc;
    using NotifyDesc = TransferMetadata::NotifyDesc;

    struct TransferRequest {
        enum OpCode { READ, WRITE };

        static constexpr uint64_t kNoTaskGroup = 0;

        OpCode opcode;
        void *source;
        SegmentID target_id;
        uint64_t target_offset;
        size_t length;
        int advise_retry_cnt = 0;
        // Per-request transport pin, TENT only.
        int transport_hint = 0;
        // Adjacent requests in a group may be scheduled as one unit.
        uint64_t task_group_id = kNoTaskGroup;
    };

    enum TransferStatusEnum {
        WAITING,
        PENDING,
        INVALID,
        CANCELED,
        COMPLETED,
        TIMEOUT,
        FAILED
    };

    struct TransferStatus {
        TransferStatusEnum s;
        size_t transferred_bytes;
    };

    struct NicLoadStats {
        std::string device_name;
        uint64_t inflight_bytes{0};
        double ewma_bandwidth_bps{0.0};
    };

    struct BatchDesc;
    struct TransferTask;

    // Limits that apply to every transfer of a batch, for the transports
    // that support them (TCP; the nvlink_proxy transport passes them to its
    // TCP fallback). Thread-safe.
    // - deadline: an absolute steady_clock time (nanoseconds since the
    //   clock's epoch, as std::chrono::steady_clock / CLOCK_MONOTONIC); 0 =
    //   none. Work not done by then fails (queued work is never started, a
    //   session in flight is cut, and a receiver that speaks TCP protocol v3
    //   applies no payload chunk that arrives after it). Set it before the
    //   transfers are submitted.
    // - cancel(): work not done yet fails at once; sessions in flight are
    //   cut. Idempotent.
    // A cut transfer fails like any other failed transfer (FAILED).
    // - unconfirmed(): attempts that ended on this side without proof that
    //   the receiver will not apply more of them (e.g. a TCP write cut
    //   mid-body before its receiver confirmed the cancel, or a GPU copy
    //   whose daemon did not answer). Shared, so that a confirmation that
    //   arrives late never touches a freed batch. The batch is quiesced --
    //   nothing of it can still be applied anywhere -- once every task is
    //   terminal and this count is 0.
    class BatchControl {
       public:
        BatchControl()
            : unconfirmed_(std::make_shared<std::atomic<int64_t>>(0)) {}
        const std::shared_ptr<std::atomic<int64_t>> &unconfirmed() const {
            return unconfirmed_;
        }
        // Counts this batch's unconfirmed attempts in |shared| (e.g. a
        // private batch run on behalf of another batch's tasks). Call before
        // submitting.
        void shareUnconfirmed(std::shared_ptr<std::atomic<int64_t>> shared) {
            if (shared) unconfirmed_ = std::move(shared);
        }
        void setDeadlineNs(int64_t deadline_ns) {
            deadline_ns_.store(deadline_ns, std::memory_order_release);
        }
        int64_t deadlineNs() const {
            return deadline_ns_.load(std::memory_order_acquire);
        }
        bool canceled() const {
            return canceled_.load(std::memory_order_acquire);
        }
        // True when work of the batch must not start (or continue) at
        // |now_ns| (steady_clock nanoseconds).
        bool expired(int64_t now_ns) const {
            if (canceled()) return true;
            const int64_t deadline = deadlineNs();
            return deadline != 0 && now_ns >= deadline;
        }
        static int64_t nowNs() {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        }
        // Registers |abort| (it cuts one piece of work in flight) to run
        // once on cancel(). Returns its id for removeAborter(), or 0 when the
        // batch is already canceled: |abort| is then not kept and the caller
        // cuts the work itself.
        uint64_t addAborter(std::function<void()> abort) {
            std::lock_guard<std::mutex> lock(mu_);
            if (canceled()) return 0;
            const uint64_t id = next_id_++;
            aborters_.emplace(id, std::move(abort));
            return id;
        }
        void removeAborter(uint64_t id) {
            if (id == 0) return;
            std::lock_guard<std::mutex> lock(mu_);
            aborters_.erase(id);
        }
        void cancel() {
            std::unordered_map<uint64_t, std::function<void()>> run;
            {
                std::lock_guard<std::mutex> lock(mu_);
                canceled_.store(true, std::memory_order_release);
                run.swap(aborters_);
            }
            for (auto &kv : run) {
                try {
                    kv.second();
                } catch (...) {
                }
            }
        }

       private:
        std::shared_ptr<std::atomic<int64_t>> unconfirmed_;
        std::atomic<int64_t> deadline_ns_{0};
        std::atomic<bool> canceled_{false};
        std::mutex mu_;
        uint64_t next_id_ = 1;
        std::unordered_map<uint64_t, std::function<void()>> aborters_;
    };

    // Progress of a batch: a task is terminal once every slice it issued
    // completed or failed (a task the transport issued no slice for counts
    // as failed). |all_terminal|: no task is in flight on this side
    // (pending == 0). |unconfirmed|: attempts that ended here without the
    // receiver confirming it applies nothing more (BatchControl).
    // |quiesced|: all_terminal and unconfirmed == 0 -- no task of the batch,
    // of any attempt, can still write anywhere.
    struct BatchProgress {
        size_t total = 0;
        size_t completed = 0;
        size_t failed = 0;
        size_t pending = 0;
        int64_t unconfirmed = 0;
        bool all_terminal = false;
        bool quiesced = false;
    };

    // NOTE ABOUT BatchID → BatchDesc conversion:
    //
    // BatchID is an opaque 64‑bit unsigned integer that carries a
    // BatchDesc pointer value. For performance reasons, this helper
    // reinterprets the integral handle directly as a BatchDesc
    // reference.
    //
    // The conversion intentionally bypasses any map or lookup to
    // minimize overhead on hot paths. The caller must ensure that
    // the underlying BatchDesc object remains alive and valid for
    // as long as the handle is in use.
    static inline BatchDesc &toBatchDesc(BatchID id) {
        return *reinterpret_cast<BatchDesc *>(id);
    }

    // Slice must be allocated on heap, as it will delete self on markSuccess
    // or markFailed.
    struct Slice {
        enum SliceStatus { PENDING, POSTED, SUCCESS, TIMEOUT, FAILED };

        void *source_addr;
        size_t length;
        TransferRequest::OpCode opcode;
        SegmentID target_id;
        std::string peer_nic_path;
        std::string source_location;
        SliceStatus status;
        TransferTask *task;
        // EFA/CXI's libfabric MR keys are 64-bit (fi_mr_key()); RDMA verbs keys
        // are 32-bit. Use a scoped alias so the width is defined in one place.
#if defined(USE_EFA) || defined(USE_CXI)
        using mr_key_t = uint64_t;
#else
        using mr_key_t = uint32_t;
#endif
        std::vector<mr_key_t> dest_rkeys;
        bool from_cache;

        // Optional resource cleanup invoked exactly once before the slice is
        // deleted or returned to the thread-local cache. The callback must not
        // delete the slice.
        using CleanupCallback = void (*)(Slice *);
        CleanupCallback cleanup_callback = nullptr;

        union {
            struct {
                uint64_t dest_addr;
                mr_key_t source_lkey;
                mr_key_t dest_rkey;
                int lkey_index;
                int rkey_index;
                std::atomic<int> *qp_depth;
                uint32_t retry_cnt;
                uint32_t max_retry_cnt;
                RdmaEndPoint *endpoint;  // Endpoint used for this transfer
            } rdma;
            struct {
                uint64_t dest_addr;
                volatile int *jetty_depth;
                uint32_t retry_cnt;
                uint32_t max_retry_cnt;
                void *r_seg;
                void *l_seg;
                void *endpoint;
            } ub;
            struct {
                void *dest_addr;
                void *cuda_stream;  // cudaStream_t, used by async NVLink
                                    // transport
                int cuda_device;    // device that owns cuda_stream
            } local;
            struct {
                void *event;  // cudaEvent_t
                int device_id;
            } nccl;
            struct {
                uint64_t dest_addr;
            } tcp;
            struct {
                uint64_t offset;
                int cufile_desc;
                uint64_t start;
                const char *file_path;
            } nvmeof;
            struct {
                void *dest_addr;
            } cxl;
            struct {
                uint64_t dest_addr;
            } hccl;
            struct {
                uint64_t dest_addr;
                void *handle;
                int64_t start_time;
                int32_t engine_id;
            } ascend_direct;
            struct {
                uint64_t dest_addr;
            } ubshmem;
            struct {
                uint64_t dest_offset;
            } flagcx;
        };

       public:
        void markSuccess() {
            status = Slice::SUCCESS;
            __atomic_fetch_add(&task->transferred_bytes, length,
                               __ATOMIC_RELAXED);
            __atomic_fetch_add(&task->success_slice_count, 1, __ATOMIC_ACQ_REL);

            check_batch_completion(task, false);
        }

        void markFailed() {
            status = Slice::FAILED;
            __atomic_fetch_add(&task->failed_slice_count, 1, __ATOMIC_ACQ_REL);

            check_batch_completion(task, true);
        }

#ifdef USE_EVENT_DRIVEN_COMPLETION
        static void sealTaskSubmission(TransferTask *task) {
            __atomic_store_n(&task->submission_sealed, true, __ATOMIC_RELEASE);
            check_batch_completion(task, false, false);
        }
#endif

        volatile int64_t ts;

       private:
        static inline void check_batch_completion(TransferTask *task,
                                                  bool is_failed,
                                                  bool count_slice = true) {
#ifdef USE_EVENT_DRIVEN_COMPLETION
            auto &batch_desc = toBatchDesc(task->batch_id);
            if (is_failed) {
                batch_desc.has_failure.store(true, std::memory_order_relaxed);
            }

            uint64_t completed =
                count_slice ? __atomic_add_fetch(&task->completed_slice_count,
                                                 1, __ATOMIC_ACQ_REL)
                            : __atomic_load_n(&task->completed_slice_count,
                                              __ATOMIC_ACQUIRE);
            if (__atomic_load_n(&task->submission_sealed, __ATOMIC_ACQUIRE) &&
                completed ==
                    __atomic_load_n(&task->slice_count, __ATOMIC_ACQUIRE) &&
                !__atomic_exchange_n(&task->completion_published, true,
                                     __ATOMIC_ACQ_REL)) {
                __atomic_store_n(&task->is_finished, true, __ATOMIC_RELAXED);

                // Increment the number of finished tasks in the batch
                // (relaxed). This counter does not itself publish data; only
                // the thread that observes the last task completion performs
                // the release-store on batch_desc.is_finished below. The waiter
                // pairs this with an acquire load, which makes all prior writes
                // (including relaxed increments) visible.
                //
                // check if this is the last task in the batch
                auto prev = batch_desc.finished_task_count.fetch_add(
                    1, std::memory_order_relaxed);

                // Last task in the batch: wake up waiting thread directly
                if (prev + 1 == batch_desc.batch_size) {
                    // Publish completion of the entire batch under the same
                    // mutex used by the waiter to avoid lost notifications.
                    //
                    // Keep a release-store because the reader has a fast path
                    // that may observe completion without taking the mutex. The
                    // acquire load in that fast path pairs with this release to
                    // make all prior updates visible. For the predicate checked
                    // under the mutex, relaxed would suffice since the mutex
                    // acquire provides the necessary visibility.
                    {
                        std::lock_guard<std::mutex> lock(
                            batch_desc.completion_mutex);
                        batch_desc.is_finished.store(true,
                                                     std::memory_order_release);
                    }
                    // Notify after releasing the lock to avoid waking threads
                    // only to block again on the mutex.
                    batch_desc.completion_cv.notify_all();
                }
            }
#endif
        }
    };

    struct ThreadLocalSliceCache {
        ThreadLocalSliceCache() : head_(0), tail_(0) {
            lazy_delete_slices_.resize(kLazyDeleteSliceCapacity);
        }

        ~ThreadLocalSliceCache() {
            for (uint64_t i = tail_; i != head_; i++) {
                auto slice = lazy_delete_slices_[i % kLazyDeleteSliceCapacity];
                delete slice;
            }
        }

        Slice *allocate() {
            Slice *slice;

            if (head_ - tail_ == 0) {
                slice = new Slice();
                slice->from_cache = false;
            } else {
                slice = lazy_delete_slices_[tail_ % kLazyDeleteSliceCapacity];
                tail_++;
                slice->from_cache = true;
            }

            return slice;
        }

        void deallocate(Slice *slice) {
            // Clear before invoking so a cached slice cannot carry a
            // transport-specific cleanup callback into its next use.
            auto cleanup = slice->cleanup_callback;
            slice->cleanup_callback = nullptr;
            if (cleanup) cleanup(slice);

            if (head_ - tail_ == kLazyDeleteSliceCapacity) {
                delete slice;
                return;
            }
            lazy_delete_slices_[head_ % kLazyDeleteSliceCapacity] = slice;
            head_++;
        }

        const static size_t kLazyDeleteSliceCapacity = 4096;
        std::vector<Slice *> lazy_delete_slices_;
        uint64_t head_, tail_;
    };

    struct TransferTask {
        volatile uint64_t slice_count = 0;
        volatile uint64_t success_slice_count = 0;
        volatile uint64_t failed_slice_count = 0;
        volatile uint64_t transferred_bytes = 0;
        volatile bool is_finished = false;
        uint64_t total_bytes = 0;
        BatchID batch_id = 0;

        // Pointer to the transport that handles this task, set by
        // MultiTransport::submitTransfer(). Used to delegate
        // transport-specific completion polling (e.g., CUDA stream
        // query for NVLink async transfers) in getTransferStatus().
        Transport *transport_ = nullptr;

#ifdef WITH_METRICS
        std::chrono::steady_clock::time_point start_time;
#endif

#ifdef USE_EVENT_DRIVEN_COMPLETION
        volatile uint64_t completed_slice_count = 0;
        volatile bool submission_sealed = true;
        volatile bool completion_published = false;
#endif

        // record the origin request
#ifdef USE_ASCEND_HETEROGENEOUS
        // need to modify the request's source address, changing it from an NPU
        // address to a CPU address.
        TransferRequest *request = nullptr;
#else
        const TransferRequest *request = nullptr;
#endif
        size_t request_count = 1;
        // record the slice list for freeing objects
        std::vector<Slice *> slice_list;
        ~TransferTask() {
            for (auto &slice : slice_list)
                Transport::getSliceCache().deallocate(slice);
        }
    };

    struct BatchDesc {
        BatchID id;
        size_t batch_size;
        std::vector<TransferTask> task_list;
        void *context;  // for transport implementers.
        int64_t start_timestamp;
        // Deadline and cancellation of the batch's transfers.
        BatchControl control;

        // Track batch progress and notifies waiters
        std::atomic<bool> has_failure{false};
        std::atomic<bool> is_finished{
            false};  // Completion flag for wait predicate
        std::atomic<uint64_t> finished_transfer_bytes{0};

#ifdef USE_EVENT_DRIVEN_COMPLETION
        // Event-driven completion: tracks batch progress and notifies waiters
        std::atomic<uint64_t> finished_task_count{0};

        // Synchronization primitives for direct notification
        std::mutex completion_mutex;
        std::condition_variable completion_cv;
#endif
    };

   public:
    virtual ~Transport() {}

    /// @brief Create a batch with specified maximum outstanding transfers.
    virtual BatchID allocateBatchID(size_t batch_size);

    /// @brief Free an allocated batch.
    virtual Status freeBatchID(BatchID batch_id);

    /// @brief Submit a batch of transfer requests to the batch.
    /// @return The number of successfully submitted transfers on success. If
    /// that number is less than nr, errno is set.
    virtual Status submitTransfer(
        BatchID batch_id, const std::vector<TransferRequest> &entries) = 0;

    virtual Status submitTransferTask(
        const std::vector<TransferTask *> &task_list) {
        return Status::NotImplemented(
            "Transport::submitTransferTask is not implemented");
    }

    virtual Status submitTransferTaskGroup(
        const std::vector<TransferTask *> &task_list) {
        return submitTransferTask(task_list);
    }

    // Grouped transports must append slices in request order so scatter can
    // recover per-request status after a grouped task fails.
    virtual bool supportsGroupedScatter() const { return false; }

    /// @brief Get the status of a submitted transfer. This function shall not
    /// be called again after completion.
    /// @return Return 1 on completed (either success or failure); 0 if still in
    /// progress.
    virtual Status getTransferStatus(BatchID batch_id, size_t task_id,
                                     TransferStatus &status) = 0;

    /// @brief The batch was canceled (BatchControl::cancel() already ran):
    /// fail its work that is still queued in this transport.
    virtual void onBatchCanceled(BatchID batch_id) {}

    /// @brief The control of the batch |slice| belongs to, or nullptr when
    /// the slice's task has no batch.
    static BatchControl *batchControl(const Slice *slice) {
        if (!slice || !slice->task || slice->task->batch_id == 0)
            return nullptr;
        return &toBatchDesc(slice->task->batch_id).control;
    }

    std::shared_ptr<TransferMetadata> &meta() { return metadata_; }

    struct BufferEntry {
        void *addr;
        size_t length;
    };

    virtual Status OpenChannel(const std::string &segment_name, SegmentID sid) {
        return Status::OK();
    }
    virtual Status CheckStatus(SegmentID sid) { return Status::OK(); }

   protected:
    virtual int install(std::string &local_server_name,
                        std::shared_ptr<TransferMetadata> meta,
                        std::shared_ptr<Topology> topo);

    std::string local_server_name_;
    std::shared_ptr<TransferMetadata> metadata_;

    RWSpinlock batch_desc_lock_;
    std::unordered_map<BatchID, std::shared_ptr<BatchDesc>> batch_desc_set_;

    static ThreadLocalSliceCache &getSliceCache();

   private:
    virtual int registerLocalMemory(void *addr, size_t length,
                                    const std::string &location,
                                    bool remote_accessible,
                                    bool update_metadata = true) = 0;

    virtual int unregisterLocalMemory(void *addr,
                                      bool update_metadata = true) = 0;

    virtual int registerLocalMemoryBatch(
        const std::vector<BufferEntry> &buffer_list,
        const std::string &location) = 0;

    virtual int unregisterLocalMemoryBatch(
        const std::vector<void *> &addr_list) = 0;

    virtual const char *getName() const = 0;
};
}  // namespace mooncake

#endif  // TRANSPORT_H_
