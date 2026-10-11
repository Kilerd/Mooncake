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
// restart can keep the new engine from allocating its memory. The daemon
// therefore tracks every mapping it opened, per (GPU context, handle), with
// the registrations that use it, and closes it as soon as the last of them
// is gone -- the engine unregistered the block, re-registered it with
// another handle, or its control connection closed -- and the copies in
// flight that still reference it have finished.
//
// The driver itself counts opens of a handle per context: opening a handle
// already open in the current context returns the same pointer and
// increments a reference count, and cudaIpcCloseMemHandle unmaps only when
// that count reaches 0 (CUDA runtime API reference, cudaIpcOpenMemHandle /
// cudaIpcCloseMemHandle). The daemon still opens a (context, handle) once
// and counts its users itself, so that it knows who holds a mapping.
//
// A mapping belongs to the engine processes that registered it. Engines are
// told apart by their client id (random per engine instance) and, for one
// process with several engine instances, by the (pid, node) they report. A
// handle whose 64 bytes equal one already mapped is served from that mapping
// only to another client of the same process, (pid, node), while a holder is
// still connected (the same allocation registered twice). Otherwise --
// another process, or a holder that disconnected and whose copies are still
// in flight (e.g. a restarted engine whose handle bytes repeat its
// predecessor's: the driver would hand back the old mapping) -- acquire()
// reports a conflict and the copy is refused (the engine uses its base
// transport) until that mapping is closed. Nothing is keyed by address
// alone.
//
// Host-only, without CUDA: the daemon supplies the open/close calls and the
// connection state, the unit test a CPU stand-in.

#ifndef MOONCAKE_NVLINK_PROXY_IPC_IMPORTS_H_
#define MOONCAKE_NVLINK_PROXY_IPC_IMPORTS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace mooncake {
namespace nvlink_proxy {

constexpr size_t kImportHandleBytes = 64;

// The process an engine runs in, as it reports itself (HELLO): a random
// per-process token (0 from engines that do not send one), with pid and
// node.
struct ImportOwner {
    uint32_t pid = 0;
    std::string node;
    uint64_t token = 0;
    bool operator==(const ImportOwner &o) const {
        return pid == o.pid && node == o.node && token == o.token;
    }
};

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
        bool shared = false;  // joined another client's mapping
        int error = 0;        // open error (kError)
        // kConflict: a client holding the mapping, its process and whether
        // any holder is still connected.
        uint64_t other = 0;
        ImportOwner other_owner;
        bool other_connected = false;
    };

    explicit ImportTable(Opener opener = Opener()) : op_(std::move(opener)) {}

    // Maps |handle| in |dev|'s context for |client| of process |owner|: the
    // mapping it already holds, one a connected client of the same process
    // holds (connected(client id) says which clients are), or a new one.
    template <class Connected>
    Acquired acquire(uint64_t client, const ImportOwner &owner, int dev,
                     const uint8_t *handle, Connected connected) {
        Acquired a;
        const Key key = makeKey(dev, handle);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            Entry &e = it->second;
            auto u = e.users.find(client);
            if (u != e.users.end()) {
                ++u->second;
                a.status = Status::kOk;
                a.ptr = e.ptr;
                return a;
            }
            bool holder_connected = false;
            for (const auto &kv : e.users)
                holder_connected |= static_cast<bool>(connected(kv.first));
            if (e.owner == owner && holder_connected) {
                e.users[client] = 1;
                a.status = Status::kOk;
                a.ptr = e.ptr;
                a.shared = true;
                ++shares_;
                return a;
            }
            a.status = Status::kConflict;
            a.other = e.users.begin()->first;
            a.other_owner = e.owner;
            a.other_connected = holder_connected;
            ++conflicts_;
            return a;
        }
        void *ptr = nullptr;
        const int err = op_.open(dev, handle, &ptr);
        if (err != 0) {
            a.error = err;
            return a;
        }
        Entry e;
        e.owner = owner;
        e.ptr = ptr;
        e.users[client] = 1;
        map_.emplace(key, std::move(e));
        ++opens_;
        a.status = Status::kOk;
        a.ptr = ptr;
        a.opened = true;
        return a;
    }

    // One use by |client| of the mapping of |handle| in |dev|'s context is
    // gone; closes the mapping when it was the last use of any client. True
    // when it closed it.
    bool release(uint64_t client, int dev, const uint8_t *handle,
                 int *error = nullptr) {
        if (error) *error = 0;
        const Key key = makeKey(dev, handle);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) return false;
        Entry &e = it->second;
        auto u = e.users.find(client);
        if (u == e.users.end()) return false;
        if (--u->second > 0) return false;
        e.users.erase(u);
        if (!e.users.empty()) return false;
        const int err = op_.close(dev, e.ptr);
        if (error) *error = err;
        map_.erase(it);
        ++closes_;
        return true;
    }

    size_t mappings() const {
        std::lock_guard<std::mutex> lock(mu_);
        return map_.size();
    }

    // Mappings |client| uses.
    size_t mappingsOf(uint64_t client) const {
        std::lock_guard<std::mutex> lock(mu_);
        size_t n = 0;
        for (const auto &kv : map_) n += kv.second.users.count(client);
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
    uint64_t shares() const {
        std::lock_guard<std::mutex> lock(mu_);
        return shares_;
    }

    Opener &opener() { return op_; }

   private:
    using Key = std::pair<int, std::array<uint8_t, kImportHandleBytes>>;
    struct Entry {
        ImportOwner owner;  // process of the client that opened it
        void *ptr = nullptr;
        std::map<uint64_t, size_t> users;  // client -> registrations
    };

    static Key makeKey(int dev, const uint8_t *handle) {
        Key k;
        k.first = dev;
        memcpy(k.second.data(), handle, kImportHandleBytes);
        return k;
    }

    mutable std::mutex mu_;
    std::map<Key, Entry> map_;
    uint64_t opens_ = 0, closes_ = 0, conflicts_ = 0, shares_ = 0;
    Opener op_;
};

}  // namespace nvlink_proxy
}  // namespace mooncake

#endif  // MOONCAKE_NVLINK_PROXY_IPC_IMPORTS_H_
