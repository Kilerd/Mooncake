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
// the work into chunks of bounded size and keeps at most |window| of them in
// flight on each stream (lane) of the thread serving the copy. Before it
// issues a chunk it predicts when that chunk would complete: the chunks
// still in flight on the lane and the chunk itself, each at the measured
// rate (RateModel: an EWMA over completed chunks, persisted across copies,
// seeded conservatively; never faster than this copy's own progress on the
// lane). It issues the chunk only if that prediction is before |deadline|
// (the engine's give-up time minus a margin); otherwise it issues nothing
// more, drains what is in flight and the copy is refused.
//
// What this guarantees, exactly:
// - No byte of the copy lands after issuePaced returns (every lane is
//   drained first), so none lands after the daemon's reply.
// - A byte lands after |deadline| only if the chunks in flight when the last
//   one was issued (at most |window| per lane) run slower than predicted.
//   Writes after the engine gave up need that misprediction to exceed the
//   margin as well: a link that slows down abruptly mid-copy, or a cold
//   lane slower than the seed rate during its first (probe) chunk.
//
// Host-only, without CUDA: the daemon supplies the stream operations, the
// unit test a CPU stand-in.

#ifndef MOONCAKE_NVLINK_PROXY_PACED_ISSUE_H_
#define MOONCAKE_NVLINK_PROXY_PACED_ISSUE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <vector>

namespace mooncake {
namespace nvlink_proxy {

// Chunks in flight per stream: one running while the next is queued, so the
// GPU never waits for the host between chunks.
constexpr size_t kPacedWindow = 2;

// The first chunk of a kind on a GPU whose rate has not been measured yet is
// at most this large and is issued alone (a probe), so that a slow link is
// measured before much of a copy is in flight.
constexpr uint64_t kProbeBytes = 8ull << 20;

// Measured transfer rate of one kind of work (e.g. copy engines of one GPU).
// Thread-safe; one instance is shared by every copy using that kind of work.
class RateModel {
   public:
    // Until the first measurement: 1 GB/s, below every P2P or host path the
    // daemon serves on healthy hardware.
    static constexpr double kSeedPsPerByte = 1000.0;
    // Fixed cost of a chunk (submission, launch, completion) in microseconds.
    static constexpr double kOverheadUs = 20.0;
    // Samples of fewer bytes are dominated by the fixed cost: they are
    // pooled until they add up to this much, then used as one.
    static constexpr uint64_t kMinSampleBytes = 256ull << 10;

    bool warm() const {
        std::lock_guard<std::mutex> lock(mu_);
        return warm_;
    }
    double psPerByte() const {
        std::lock_guard<std::mutex> lock(mu_);
        return ps_per_byte_;
    }
    // GB/s of the current estimate (stats).
    double gbps() const { return 1000.0 / psPerByte(); }

    // |bytes| took |us| (from start to completion). The first sample replaces
    // the seed; later ones move the estimate by 1/2 when slower (a slowdown
    // is followed at once) and by 1/8 when faster. Small samples are pooled
    // (so a model that only ever sees small chunks still warms up).
    void observe(uint64_t bytes, double us) {
        if (bytes == 0 || us <= 0) return;
        std::lock_guard<std::mutex> lock(mu_);
        double ps;
        if (bytes < kMinSampleBytes) {
            pooled_bytes_ += bytes;
            pooled_us_ += us;
            ++pooled_count_;
            if (pooled_bytes_ < kMinSampleBytes) return;
            const double total_us = pooled_us_;
            const double fixed_us = pooled_count_ * kOverheadUs;
            const double pooled = static_cast<double>(pooled_bytes_);
            pooled_bytes_ = 0;
            pooled_us_ = 0;
            pooled_count_ = 0;
            // All of it fixed cost by the model: take the whole time (slower
            // than any real rate, so on the safe side).
            ps = (total_us > fixed_us ? total_us - fixed_us : total_us) * 1e6 /
                 pooled;
        } else {
            ps = std::max(0.0, us - kOverheadUs) * 1e6 /
                 static_cast<double>(bytes);
            if (ps <= 0) return;
        }
        if (!warm_) {
            ps_per_byte_ = ps;
            warm_ = true;
            return;
        }
        const double alpha = ps > ps_per_byte_ ? 0.5 : 0.125;
        ps_per_byte_ += alpha * (ps - ps_per_byte_);
    }

   private:
    mutable std::mutex mu_;
    double ps_per_byte_ = kSeedPsPerByte;
    bool warm_ = false;
    uint64_t pooled_bytes_ = 0;
    double pooled_us_ = 0;
    uint64_t pooled_count_ = 0;
};

// Predicted run time of |bytes| at |ps_per_byte|, in microseconds.
inline double predictUs(uint64_t bytes, double ps_per_byte) {
    return RateModel::kOverheadUs +
           static_cast<double>(bytes) * ps_per_byte / 1e6;
}

// Consecutive entries [begin, end) issued together, |bytes| in total.
struct ChunkSpan {
    size_t begin = 0;
    size_t end = 0;
    uint64_t bytes = 0;
};

// Splits entries of lengths |len[0..n)| into consecutive spans of at most
// |max_entries| entries and at most |max_bytes| bytes, the first span at
// most |first_max_bytes| bytes when that is not 0 (a probe); an entry longer
// than the limit is a span of its own (split such entries first). 0 means no
// limit.
inline std::vector<ChunkSpan> planChunks(const uint64_t *len, size_t n,
                                         size_t max_entries, uint64_t max_bytes,
                                         uint64_t first_max_bytes = 0) {
    std::vector<ChunkSpan> out;
    ChunkSpan cur;
    for (size_t i = 0; i < n; ++i) {
        uint64_t cap = max_bytes;
        if (out.empty() && first_max_bytes)
            cap = max_bytes ? std::min(max_bytes, first_max_bytes)
                            : first_max_bytes;
        const bool full =
            cur.end > cur.begin &&
            ((max_entries && cur.end - cur.begin >= max_entries) ||
             (cap && cur.bytes + len[i] > cap));
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

// What issuePaced needs to know about a chunk.
struct PacedChunk {
    int lane = 0;  // the stream it runs on
    uint64_t bytes = 0;
    RateModel *model = nullptr;  // its kind of work (nullptr: seed rate)
};

struct PacedResult {
    size_t issued = 0;        // chunks issued (all of them on success)
    bool deadline = false;    // stopped: a chunk would not finish in time
    double predicted_us = 0;  // the prediction that stopped it
    int error = 0;            // first error of issue/wait (0: none)
    int drain_error = 0;      // first error of the final drain (0: none)
};

// Issues chunks[0..n) in order. Before chunk k on its lane:
// - room is made: the lane's oldest chunk is waited for while |window|
//   chunks are in flight, or while a chunk of a cold model (no measurement
//   yet) is in flight or about to be issued (a cold chunk runs alone);
// - its completion is predicted: now + the chunks in flight on the lane +
//   chunk k, each at max(its model's rate, the lane's progress in this copy);
//   past |deadline_us| nothing more is issued;
// - issue(k, slot) submits it and marks it done in |slot| (< window).
// Every completed chunk feeds its model (time from its start -- issue, or
// the previous completion on the lane -- to its observed completion).
// drain(lane) runs for every lane at the end, in every case, so nothing
// issued here is still running when this returns. Callbacks return 0 or an
// error code: issue(size_t k, size_t slot), wait(int lane, size_t slot),
// drain(int lane); now() returns microseconds.
template <class Now, class Issue, class Wait, class Drain>
PacedResult issuePaced(const std::vector<PacedChunk> &chunks, size_t window,
                       uint64_t deadline_us, Now now, Issue issue, Wait wait,
                       Drain drain) {
    PacedResult r;
    if (window == 0) window = 1;
    struct Flight {
        size_t k;
        size_t slot;
        uint64_t issued_us;
        bool cold;
    };
    struct Lane {
        std::deque<Flight> q;
        size_t count = 0;           // chunks issued
        uint64_t first_us = 0;      // first issue
        uint64_t last_done_us = 0;  // last observed completion
        uint64_t done_bytes = 0;    // bytes observed complete
    };
    std::map<int, Lane> lanes;
    auto modelPs = [&](size_t k) {
        return chunks[k].model ? chunks[k].model->psPerByte()
                               : RateModel::kSeedPsPerByte;
    };
    // This copy's own progress on the lane, as a rate floor.
    auto lanePs = [&](const Lane &ln) {
        if (ln.done_bytes < RateModel::kMinSampleBytes) return 0.0;
        return static_cast<double>(ln.last_done_us - ln.first_us) * 1e6 /
               static_cast<double>(ln.done_bytes);
    };
    auto settle = [&](Lane &ln, const Flight &f, uint64_t t) {
        const uint64_t start = std::max(f.issued_us, ln.last_done_us);
        if (chunks[f.k].model)
            chunks[f.k].model->observe(chunks[f.k].bytes,
                                       static_cast<double>(t - start));
        ln.done_bytes += chunks[f.k].bytes;
        ln.last_done_us = t;
    };
    for (size_t k = 0; k < chunks.size(); ++k) {
        Lane &ln = lanes[chunks[k].lane];
        const bool cold = chunks[k].model && !chunks[k].model->warm();
        bool failed = false;
        while (!ln.q.empty()) {
            bool cold_in_flight = false;
            for (const Flight &f : ln.q) cold_in_flight |= f.cold;
            if (ln.q.size() < window && !cold && !cold_in_flight) break;
            const Flight f = ln.q.front();
            const int e = wait(chunks[k].lane, f.slot);
            if (e) {
                r.error = e;
                failed = true;
                break;
            }
            settle(ln, f, now());
            ln.q.pop_front();
        }
        if (failed) break;
        const double floor_ps = lanePs(ln);
        double us = predictUs(chunks[k].bytes, std::max(modelPs(k), floor_ps));
        for (const Flight &f : ln.q)
            us +=
                predictUs(chunks[f.k].bytes, std::max(modelPs(f.k), floor_ps));
        const uint64_t t = now();
        if (static_cast<double>(t) + us > static_cast<double>(deadline_us)) {
            r.deadline = true;
            r.predicted_us = us;
            break;
        }
        const size_t slot = ln.count % window;
        const int e = issue(k, slot);
        if (e) {
            r.error = e;
            break;
        }
        if (ln.count == 0) ln.first_us = t;
        ++ln.count;
        ++r.issued;
        ln.q.push_back({k, slot, t, cold});
    }
    for (auto &kv : lanes) {
        Lane &ln = kv.second;
        const int e = drain(kv.first);
        if (e && !r.drain_error) r.drain_error = e;
        if (e || r.error || ln.q.empty()) continue;
        // The chunks still in flight ran back to back: one sample for
        // them when they share a model.
        const uint64_t t = now();
        const uint64_t start =
            std::max(ln.q.front().issued_us, ln.last_done_us);
        RateModel *m = chunks[ln.q.front().k].model;
        uint64_t bytes = 0;
        for (const Flight &f : ln.q) {
            bytes += chunks[f.k].bytes;
            if (chunks[f.k].model != m) m = nullptr;
        }
        if (m) m->observe(bytes, static_cast<double>(t - start));
    }
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
