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

// The legacy CUDA IPC mappings imported by mooncake_nvlink_proxy.
//
// An imported mapping keeps the exporting process's device memory allocated
// even after that process died, so a mapping left open after an engine
// restart can keep the new engine from allocating its memory. A process also
// holds at most one mapping of an allocation per context: opening a handle
// again where it is already open fails or returns the same mapping, and one
// close unmaps it for every user (a registration closing "its" mapping would
// pull it from under another). The daemon therefore opens a handle once per
// (GPU context, handle), counts the registrations that use it and
// closes it as soon as the last of them is gone -- the engine unregistered
// the block, re-registered it with another handle, or its control
// connection closed -- and the copies in flight that still reference it
// have finished.
//
// Every mapping belongs to one engine: the client id, random per engine
// process. A handle whose 64 bytes equal one still mapped for another client
// (another engine process: handles carry nothing the daemon could check) is
// never served from that mapping: acquire() reports the conflict and the
// copy is refused (the engine uses its base transport) until the other
// client's mapping is closed. Nothing is keyed by address alone.
//
// Host-only, without CUDA: the daemon supplies the open/close calls, the unit
// test a CPU stand-in.

#ifndef MOONCAKE_NVLINK_PROXY_IPC_IMPORTS_H_
#define MOONCAKE_NVLINK_PROXY_IPC_IMPORTS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>

namespace mooncake {
namespace nvlink_proxy {

constexpr size_t kImportHandleBytes = 64;

// Opener: int open(int dev, const uint8_t *handle, void **ptr) and
// int close(int dev, void *ptr), returning 0 or an error code.
template <class Opener>
class ImportTable {
   public:
    enum class Status { kOk, kConflict, kError };

    struct Acquired {
        Status status = Status::kError;
        void *ptr = nullptr;
        bool opened = false;  // opened now (not shared)
        int error = 0;        // open error (kError)
        uint64_t other = 0;   // client holding the handle (kConflict)
    };

    explicit ImportTable(Opener opener = Opener()) : op_(std::move(opener)) {}

    // Maps |handle| in |dev|'s context for |client|: the mapping the client
    // already holds (one more user) or a new one.
    Acquired acquire(uint64_t client, int dev, const uint8_t *handle) {
        Acquired a;
        const Key key = makeKey(dev, handle);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            if (it->second.client != client) {
                a.status = Status::kConflict;
                a.other = it->second.client;
                ++conflicts_;
                return a;
            }
            ++it->second.users;
            a.status = Status::kOk;
            a.ptr = it->second.ptr;
            return a;
        }
        void *ptr = nullptr;
        const int e = op_.open(dev, handle, &ptr);
        if (e != 0) {
            a.error = e;
            return a;
        }
        map_.emplace(key, Entry{client, ptr, 1});
        ++opens_;
        a.status = Status::kOk;
        a.ptr = ptr;
        a.opened = true;
        return a;
    }

    // One user of |client|'s mapping of |handle| in |dev|'s context is gone;
    // closes the mapping when it was the last. True when it closed it.
    bool release(uint64_t client, int dev, const uint8_t *handle,
                 int *error = nullptr) {
        if (error) *error = 0;
        const Key key = makeKey(dev, handle);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(key);
        if (it == map_.end() || it->second.client != client) return false;
        if (--it->second.users > 0) return false;
        const int e = op_.close(dev, it->second.ptr);
        if (error) *error = e;
        map_.erase(it);
        ++closes_;
        return true;
    }

    size_t mappings() const {
        std::lock_guard<std::mutex> lock(mu_);
        return map_.size();
    }

    size_t mappingsOf(uint64_t client) const {
        std::lock_guard<std::mutex> lock(mu_);
        size_t n = 0;
        for (const auto &kv : map_) n += kv.second.client == client;
        return n;
    }

    uint64_t opens() const {
        std::lock_guard<std::mutex> lock(mu_);
        return opens_;
    }
    uint64_t closes() const {
        std::lock_guard<std::mutex> lock(mu_);
        return closes_;
    }
    uint64_t conflicts() const {
        std::lock_guard<std::mutex> lock(mu_);
        return conflicts_;
    }

    Opener &opener() { return op_; }

   private:
    using Key = std::pair<int, std::array<uint8_t, kImportHandleBytes>>;
    struct Entry {
        uint64_t client;
        void *ptr;
        size_t users;
    };

    static Key makeKey(int dev, const uint8_t *handle) {
        Key k;
        k.first = dev;
        memcpy(k.second.data(), handle, kImportHandleBytes);
        return k;
    }

    mutable std::mutex mu_;
    std::map<Key, Entry> map_;
    uint64_t opens_ = 0, closes_ = 0, conflicts_ = 0;
    Opener op_;
};

}  // namespace nvlink_proxy
}  // namespace mooncake

#endif  // MOONCAKE_NVLINK_PROXY_IPC_IMPORTS_H_
