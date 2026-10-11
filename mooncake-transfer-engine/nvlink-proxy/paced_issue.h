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

// Paced issue of the direct (peer-to-peer) part of a COPY.
//
// The engine gives up on a copy at a deadline and rewrites it over its base
// transport; a write of the daemon that lands after that may overwrite data
// the engine has moved on to (a recycled page). The daemon therefore splits
// the work into chunks of bounded size, keeps at most |window| chunks in
// flight per GPU stream (lane), and checks the copy's hard deadline before it
// issues each chunk. Once the deadline has passed nothing more is issued,
// what is in flight (at most |window| chunks per lane) is drained, and the
// copy is refused: no byte lands after the reply, and the reply leaves at
// most the time of |window| chunks after the deadline.
//
// Host-only, without CUDA: the daemon supplies the stream operations, the
// unit test a CPU stand-in.

#ifndef MOONCAKE_NVLINK_PROXY_PACED_ISSUE_H_
#define MOONCAKE_NVLINK_PROXY_PACED_ISSUE_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace mooncake {
namespace nvlink_proxy {

// Chunks in flight per GPU stream: one running while the next is queued, so
// the GPU never waits for the host between chunks.
constexpr size_t kPacedWindow = 2;

// Consecutive entries [begin, end) issued together, |bytes| in total.
struct ChunkSpan {
    size_t begin = 0;
    size_t end = 0;
    uint64_t bytes = 0;
};

// Splits entries of lengths |len[0..n)| into consecutive spans of at most
// |max_entries| entries and at most |max_bytes| bytes; an entry longer than
// |max_bytes| is a span of its own (split such entries first, see
// splitLength). max_entries and max_bytes of 0 mean no limit.
inline std::vector<ChunkSpan> planChunks(const uint64_t *len, size_t n,
                                         size_t max_entries,
                                         uint64_t max_bytes) {
    std::vector<ChunkSpan> out;
    ChunkSpan cur;
    for (size_t i = 0; i < n; ++i) {
        const bool full =
            cur.end > cur.begin &&
            ((max_entries && cur.end - cur.begin >= max_entries) ||
             (max_bytes && cur.bytes + len[i] > max_bytes));
        if (full) {
            out.push_back(cur);
            cur = ChunkSpan{i, i, 0};
        }
        cur.end = i + 1;
        cur.bytes += len[i];
    }
    if (cur.end > cur.begin) out.push_back(cur);
    return out;
}

// Number of pieces of at most |max_bytes| a copy of |len| bytes is split into
// (1 when max_bytes is 0 or len fits).
inline uint64_t splitLength(uint64_t len, uint64_t max_bytes) {
    if (!max_bytes || len <= max_bytes) return 1;
    return (len + max_bytes - 1) / max_bytes;
}

struct PacedResult {
    size_t issued = 0;      // chunks issued (all of them on success)
    bool deadline = false;  // stopped because the deadline had passed
    int error = 0;          // first error of issue/record/wait (0: none)
    int drain_error = 0;    // error of the final drain (0: none)
};

// Issues chunks 0..lanes.size()-1 in order; chunk k runs on lane lanes[k].
// Before chunk number c of a lane (counted per lane) is issued, the lane's
// chunk c - window must have completed (wait(lane, c % window)); then the
// deadline is checked (now() >= deadline_us stops), then issue(k) and
// record(lane, c % window) mark the chunk. drain() runs at the end in every
// case, so nothing issued here is still running when this returns. The
// callbacks return 0 or an error code.
template <class Now, class Issue, class Record, class Wait, class Drain>
PacedResult issuePaced(const std::vector<int> &lanes, size_t window,
                       uint64_t deadline_us, Now now, Issue issue,
                       Record record, Wait wait, Drain drain) {
    PacedResult r;
    if (window == 0) window = 1;
    std::map<int, size_t> per_lane;  // chunks issued per lane
    for (size_t k = 0; k < lanes.size(); ++k) {
        const int lane = lanes[k];
        size_t &c = per_lane[lane];
        if (c >= window) {
            const int e = wait(lane, c % window);
            if (e) {
                r.error = e;
                break;
            }
        }
        if (now() >= deadline_us) {
            r.deadline = true;
            break;
        }
        int e = issue(k);
        if (e) {
            r.error = e;
            break;
        }
        ++r.issued;
        e = record(lane, c % window);
        if (e) {
            r.error = e;
            break;
        }
        ++c;
    }
    r.drain_error = drain();
    return r;
}

// Order of issue for per-lane chunk lists: round robin over the lanes, so
// that a copy with work on two GPUs keeps both busy. |counts| maps lane ->
// number of chunks; the result holds (lane, index within the lane) pairs.
inline std::vector<std::pair<int, size_t>> interleaveLanes(
    const std::map<int, size_t> &counts) {
    std::vector<std::pair<int, size_t>> order;
    size_t total = 0, most = 0;
    for (const auto &kv : counts) {
        total += kv.second;
        if (kv.second > most) most = kv.second;
    }
    order.reserve(total);
    for (size_t i = 0; i < most; ++i)
        for (const auto &kv : counts)
            if (i < kv.second) order.push_back({kv.first, i});
    return order;
}

}  // namespace nvlink_proxy
}  // namespace mooncake

#endif  // MOONCAKE_NVLINK_PROXY_PACED_ISSUE_H_
