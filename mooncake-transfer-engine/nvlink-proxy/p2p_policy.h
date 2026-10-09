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

// P2P policy of mooncake_nvlink_proxy: which GPU pairs the daemon may move
// data between peer to peer.
//
// Some platforms report peer access between two GPUs (cudaDeviceCanAccessPeer,
// nvidia-smi topo -p2p) and complete peer copies without any error, yet
// deliver corrupt data. The daemon therefore copies a known pattern between
// both GPUs of a pair before it serves copies between them (the P2P
// self-test) and denies a pair that fails. A copy between the GPUs of a
// denied pair is refused with kNoPeerAccess, which the engines answer by
// moving the request to their base transport (RDMA / TCP).
//
// Everything in this header is host-only, without CUDA, so that it can be
// unit-tested without GPUs.

#ifndef MOONCAKE_NVLINK_PROXY_P2P_POLICY_H_
#define MOONCAKE_NVLINK_PROXY_P2P_POLICY_H_

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mooncake {
namespace nvlink_proxy {

// --p2p-selftest: enforce (deny a pair that fails or cannot be tested), warn
// (log only, keep using P2P) or off (no test).
enum class SelfTestMode { kEnforce, kWarn, kOff };

inline bool parseSelfTestMode(const std::string &s, SelfTestMode &out) {
    if (s == "enforce") {
        out = SelfTestMode::kEnforce;
    } else if (s == "warn") {
        out = SelfTestMode::kWarn;
    } else if (s == "off") {
        out = SelfTestMode::kOff;
    } else {
        return false;
    }
    return true;
}

inline const char *selfTestModeName(SelfTestMode m) {
    switch (m) {
        case SelfTestMode::kEnforce:
            return "enforce";
        case SelfTestMode::kWarn:
            return "warn";
        case SelfTestMode::kOff:
            return "off";
    }
    return "?";
}

// -------------------------------------------------------------- PCI address
// domain:bus:device of a GPU (the function is always 0 for a GPU and is not
// compared).
struct PciAddr {
    uint32_t domain = 0;
    uint32_t bus = 0;
    uint32_t device = 0;
};

inline bool operator==(const PciAddr &a, const PciAddr &b) {
    return a.domain == b.domain && a.bus == b.bus && a.device == b.device;
}

namespace detail {
inline bool parseHex(const std::string &s, size_t max_digits, uint32_t max,
                     uint32_t &out) {
    if (s.empty() || s.size() > max_digits) return false;
    uint64_t v = 0;
    for (char c : s) {
        if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
        v = v * 16 +
            static_cast<uint64_t>(
                std::isdigit(static_cast<unsigned char>(c))
                    ? c - '0'
                    : std::tolower(static_cast<unsigned char>(c)) - 'a' + 10);
    }
    if (v > max) return false;
    out = static_cast<uint32_t>(v);
    return true;
}
}  // namespace detail

// Parses "[domain:]bus:device[.function]" (hex), the forms printed by
// nvidia-smi ("00000000:3b:00.0"), lspci ("3b:00.0") and this daemon
// ("0000:3b:00"). Surrounding blanks are ignored.
inline bool parsePciAddr(const std::string &in, PciAddr &out) {
    size_t b = 0, e = in.size();
    while (b < e && std::isspace(static_cast<unsigned char>(in[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(in[e - 1]))) --e;
    std::string s = in.substr(b, e - b);
    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
        size_t colon = s.find(':', start);
        parts.push_back(s.substr(start, colon - start));
        if (colon == std::string::npos) break;
        start = colon + 1;
    }
    if (parts.size() != 2 && parts.size() != 3) return false;
    std::string dev = parts.back();
    size_t dot = dev.find('.');
    if (dot != std::string::npos) {
        uint32_t fn = 0;
        if (!detail::parseHex(dev.substr(dot + 1), 1, 7, fn)) return false;
        dev = dev.substr(0, dot);
    }
    PciAddr a;
    if (parts.size() == 3 &&
        !detail::parseHex(parts[0], 8, 0xffffffffu, a.domain))
        return false;
    if (!detail::parseHex(parts[parts.size() - 2], 2, 0xff, a.bus))
        return false;
    if (!detail::parseHex(dev, 2, 0x1f, a.device)) return false;
    out = a;
    return true;
}

// "0000:3b:00"
inline std::string formatPciAddr(const PciAddr &a) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%04x:%02x:%02x", a.domain, a.bus, a.device);
    return buf;
}

// "3b:00" in domain 0, else the full form (for compact lists).
inline std::string formatPciAddrShort(const PciAddr &a) {
    if (a.domain != 0) return formatPciAddr(a);
    char buf[16];
    snprintf(buf, sizeof(buf), "%02x:%02x", a.bus, a.device);
    return buf;
}

// ------------------------------------------------------------- peer policy
// Ordered GPU pairs (data flows src -> dst; daemon device ordinals) the
// daemon refuses to copy between. Copy threads read it while a device that is
// being initialized writes the pairs involving that device, so the entries
// are atomics. Copies within one GPU are never denied.
class PeerPolicy {
   public:
    enum Reason : uint8_t {
        kAllowed = 0,
        kStatic = 1,          // --deny-peer / --deny-pair
        kSelfTestFailed = 2,  // the self-test read back corrupt data
        kUntested = 3,        // not (yet) verified by a self-test (enforce)
    };

    explicit PeerPolicy(int num_devices = 0) { reset(num_devices); }

    void reset(int num_devices) {
        n_ = std::max(0, num_devices);
        m_.reset(n_ ? new std::atomic<uint8_t>[size_t(n_) * n_] : nullptr);
        for (int i = 0; i < n_ * n_; ++i) m_[i].store(kAllowed);
    }

    int numDevices() const { return n_; }

    // Sets the state of a direction (kAllowed clears it). A denial from the
    // command line (kStatic) is never overridden.
    void set(int src, int dst, Reason reason) {
        if (!valid(src, dst) || src == dst) return;
        std::atomic<uint8_t> &e = m_[index(src, dst)];
        uint8_t cur = e.load();
        while (cur != kStatic && !e.compare_exchange_weak(cur, reason)) {
        }
    }

    void setPair(int a, int b, Reason reason) {
        set(a, b, reason);
        set(b, a, reason);
    }

    // Keeps the first reason a pair was denied for.
    void deny(int src, int dst, Reason reason) {
        if (!valid(src, dst) || src == dst || reason == kAllowed) return;
        uint8_t expected = kAllowed;
        m_[index(src, dst)].compare_exchange_strong(expected, reason);
    }

    // Both directions.
    void denyPair(int a, int b, Reason reason) {
        deny(a, b, reason);
        deny(b, a, reason);
    }

    // Every pair with |dev|.
    void denyDevice(int dev, Reason reason) {
        for (int other = 0; other < n_; ++other) denyPair(dev, other, reason);
    }

    bool denied(int src, int dst) const { return reason(src, dst) != kAllowed; }

    Reason reason(int src, int dst) const {
        if (!valid(src, dst) || src == dst) return kAllowed;
        return static_cast<Reason>(m_[index(src, dst)].load());
    }

    size_t deniedCount() const {
        size_t n = 0;
        for (int s = 0; s < n_; ++s)
            for (int d = 0; d < n_; ++d) n += denied(s, d) ? 1 : 0;
        return n;
    }

    // "src>dst,..." with |names|[ordinal] (or the ordinal) for every denied
    // ordered pair; empty when none is.
    std::string deniedList(const std::vector<std::string> &names = {}) const {
        std::string out;
        auto name = [&](int i) {
            return i < static_cast<int>(names.size()) ? names[i]
                                                      : std::to_string(i);
        };
        for (int s = 0; s < n_; ++s) {
            for (int d = 0; d < n_; ++d) {
                if (!denied(s, d)) continue;
                if (!out.empty()) out += ',';
                out += name(s) + ">" + name(d);
            }
        }
        return out;
    }

    static const char *reasonName(Reason r) {
        switch (r) {
            case kAllowed:
                return "allowed";
            case kStatic:
                return "denied by the command line";
            case kSelfTestFailed:
                return "P2P self-test failed";
            case kUntested:
                return "not verified by the P2P self-test (yet)";
        }
        return "?";
    }

   private:
    bool valid(int src, int dst) const {
        return src >= 0 && dst >= 0 && src < n_ && dst < n_;
    }
    size_t index(int src, int dst) const { return size_t(src) * n_ + dst; }

    int n_ = 0;
    std::unique_ptr<std::atomic<uint8_t>[]> m_;
};

// --------------------------------------------------------- self-test plan
constexpr uint64_t kSelfTestRowBytes = 2048;  // a typical KV page row

// One copy of a self-test pass, offsets into the source / destination test
// buffers.
struct TestCopy {
    uint64_t src_off = 0;
    uint64_t dst_off = 0;
    uint64_t len = 0;
};

inline uint64_t splitmix64(uint64_t &state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

inline uint64_t mixSeed(uint64_t a, uint64_t b) {
    uint64_t s = a ^ (b * 0x9e3779b97f4a7c15ULL);
    return splitmix64(s);
}

// Copies of one self-test pass over two buffers of |buf_bytes| each.
// |large| = entries the daemon gives to the copy engines (at least
// |gather_threshold| bytes: whole 2 KiB rows, a few hundred rows, an odd
// length, a multi-MiB run when it fits); otherwise entries for the gather
// kernel (single 2 KiB rows, a few KiB, odd lengths and byte-misaligned
// offsets, all below |gather_threshold|). Destination ranges never overlap
// and are separated by untouched guard bytes, so a verifier also sees stray
// writes. |iter| shifts the layout and picks other source offsets.
inline std::vector<TestCopy> selfTestPlan(uint64_t buf_bytes, uint32_t iter,
                                          bool large,
                                          uint64_t gather_threshold) {
    const uint64_t row = kSelfTestRowBytes;
    const uint64_t guard = 4096;
    std::vector<uint64_t> lens;
    std::vector<uint64_t> src_mis, dst_mis;
    if (large) {
        const uint64_t min_len =
            std::max<uint64_t>((gather_threshold + row - 1) / row * row, row);
        // First an entry of at least the gather threshold (it always fits a
        // buffer of selfTestMinBufferBytes()), then small entries (the copy
        // engines take those too when the kernel is unusable or disabled),
        // then a misaligned odd length and longer runs as they fit.
        lens = {min_len,
                row,
                1000,
                3 * row + 16,
                8192,
                min_len + 1000,
                min_len + row,
                std::max(min_len, 356 * row) + 7 * row,
                std::max(min_len, 1024 * row) + 3 * row};
        src_mis = {0, 0, 16, 1, 0, 16, 2 * row, 0, 0};
        dst_mis = {0, 0, 3, 0, 0, 3, row, 0, 0};
    } else {
        for (int k = 0; k < 12; ++k) lens.push_back(row);
        for (uint64_t len : {3 * row + 16, uint64_t(8192), uint64_t(1000),
                             uint64_t(64 * 1024), uint64_t(1)}) {
            lens.push_back(len);
        }
        lens.erase(std::remove_if(lens.begin(), lens.end(),
                                  [&](uint64_t l) {
                                      return gather_threshold == 0 ||
                                             l >= gather_threshold;
                                  }),
                   lens.end());
        src_mis = {0, 0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 0, 16, 0, 3, 0, 5};
        dst_mis = {0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 9};
    }
    std::vector<TestCopy> plan;
    uint64_t seed = mixSeed(iter + 1, large ? 0x51 : 0x4b);
    uint64_t cursor = (iter % 4) * row + guard;
    for (size_t k = 0; k < lens.size(); ++k) {
        const uint64_t len = lens[k];
        const uint64_t sm = src_mis[k % src_mis.size()];
        const uint64_t dm = dst_mis[k % dst_mis.size()];
        if (len + sm + guard > buf_bytes) continue;
        const uint64_t dst_off = cursor + dm;
        if (dst_off + len + guard > buf_bytes) continue;
        // Source: a row-aligned (plus |sm|) offset anywhere it fits; sources
        // may overlap, they are only read.
        const uint64_t slots = (buf_bytes - len - sm) / row;
        const uint64_t src_off = (splitmix64(seed) % slots) * row + sm;
        plan.push_back({src_off, dst_off, len});
        cursor = (dst_off + len + guard + row - 1) / row * row;
    }
    return plan;
}

// Smallest test buffer whose copy-engine plan still holds its first entry,
// one of at least |gather_threshold| bytes, in every iteration.
inline uint64_t selfTestMinBufferBytes(uint64_t gather_threshold) {
    const uint64_t row = kSelfTestRowBytes;
    const uint64_t min_len =
        std::max<uint64_t>((gather_threshold + row - 1) / row * row, row);
    const uint64_t need = min_len + 3 * row + 3 * 4096;  // shift and guards
    const uint64_t align = 64 * 1024;
    return std::max<uint64_t>((need + align - 1) / align * align, 1ull << 20);
}

// Deterministic pattern: one splitmix64 word per 8 bytes (|n| need not be a
// multiple of 8).
inline void fillPattern(uint8_t *buf, uint64_t n, uint64_t seed) {
    uint64_t state = seed;
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t v = splitmix64(state);
        memcpy(buf + i, &v, 8);
    }
    if (i < n) {
        uint64_t v = splitmix64(state);
        memcpy(buf + i, &v, n - i);
    }
}

// What the destination buffer must hold after |plan| copied from |src| into
// a buffer that held |poison|.
inline void expectedImage(const uint8_t *src, const uint8_t *poison, uint64_t n,
                          const std::vector<TestCopy> &plan, uint8_t *out) {
    memcpy(out, poison, n);
    for (const auto &c : plan) memcpy(out + c.dst_off, src + c.src_off, c.len);
}

struct Mismatch {
    uint64_t bytes = 0;           // bytes that differ
    uint64_t first = UINT64_MAX;  // offset of the first one
    uint8_t expected = 0, actual = 0;
    bool any() const { return bytes != 0; }
};

inline Mismatch compareImages(const uint8_t *expected, const uint8_t *actual,
                              uint64_t n) {
    Mismatch m;
    uint64_t i = 0;
    // Fast path: whole 4 KiB pieces that match.
    while (i < n) {
        const uint64_t chunk = std::min<uint64_t>(4096, n - i);
        if (memcmp(expected + i, actual + i, chunk) != 0) {
            for (uint64_t j = i; j < i + chunk; ++j) {
                if (expected[j] == actual[j]) continue;
                if (!m.any()) {
                    m.first = j;
                    m.expected = expected[j];
                    m.actual = actual[j];
                }
                ++m.bytes;
            }
        }
        i += chunk;
    }
    return m;
}

// ------------------------------------------------------- host staging pool
// Slots of the pinned pool of one NUMA node: |total_bytes| split over
// |nodes| nodes, slots of 1 to 16 MiB (an eighth of the node's share, in
// 64 KiB units). False when that leaves fewer than two slots: a copy needs
// two.
inline bool stagingGeometry(uint64_t total_bytes, size_t nodes,
                            uint64_t *slot_bytes, size_t *slots) {
    const uint64_t share = total_bytes / std::max<size_t>(1, nodes);
    uint64_t slot = std::min<uint64_t>(16ull << 20, share / 8);
    slot = std::max<uint64_t>(1ull << 20, slot / (64 << 10) * (64 << 10));
    *slot_bytes = slot;
    *slots = size_t(share / slot);
    return *slots >= 2;
}

// Offset of the next piece in a slot: 16-byte aligned (the widest access
// of the gather kernel and the copy engines' preferred alignment).
inline uint64_t alignSlotOffset(uint64_t off) {
    return (off + 15) & ~uint64_t(15);
}

// ------------------------------------------------------------- verdicts
// Outcome of testing one direction (data flows src -> dst). Only a data
// mismatch is evidence against a pair; anything that kept the test from
// running (no memory, a CUDA error, ...) leaves it untested. A direction
// without peer access is not judged: its copies are refused as no-P2P.
enum class DirVerdict { kPass, kFail, kUntested, kNoPeer };

inline const char *dirVerdictName(DirVerdict v) {
    switch (v) {
        case DirVerdict::kPass:
            return "PASS";
        case DirVerdict::kFail:
            return "FAIL";
        case DirVerdict::kUntested:
            return "UNTESTED";
        case DirVerdict::kNoPeer:
            return "NOPEER";
    }
    return "?";
}

inline bool parseDirVerdict(const std::string &s, DirVerdict &out) {
    for (DirVerdict v : {DirVerdict::kPass, DirVerdict::kFail,
                         DirVerdict::kUntested, DirVerdict::kNoPeer}) {
        if (s == dirVerdictName(v)) {
            out = v;
            return true;
        }
    }
    return false;
}

enum class PairVerdict { kPassed, kFailed, kUntested };

// A pair fails when either direction read back wrong data; otherwise it is
// untested while a direction with peer access could not be tested.
inline PairVerdict combineVerdicts(DirVerdict ab, DirVerdict ba) {
    if (ab == DirVerdict::kFail || ba == DirVerdict::kFail)
        return PairVerdict::kFailed;
    if (ab == DirVerdict::kUntested || ba == DirVerdict::kUntested)
        return PairVerdict::kUntested;
    return PairVerdict::kPassed;
}

// ------------------------------------------- self-test child process wire
// The daemon tests pairs in a child process (the same binary, re-executed)
// that reports on its stdout, one line each:
//   BEGIN <bus a> <bus b>      testing of the pair started
//   DIR <bus src> <bus dst> <PASS|FAIL|UNTESTED|NOPEER>
//   END <bus a> <bus b>        both directions reported
// A pair with BEGIN but no END when the child died or hung is failed.
using PciPair = std::pair<PciAddr, PciAddr>;

// "<bus a>+<bus b>,..." (the child's --p2p-selftest-pairs argument).
inline std::string encodePciPairs(const std::vector<PciPair> &pairs) {
    std::string out;
    for (const auto &p : pairs) {
        if (!out.empty()) out += ',';
        out += formatPciAddr(p.first) + "+" + formatPciAddr(p.second);
    }
    return out;
}

inline bool parsePciPairs(const std::string &s, std::vector<PciPair> &out) {
    out.clear();
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        std::string item = s.substr(start, comma - start);
        size_t plus = item.find('+');
        PciPair p;
        if (plus == std::string::npos ||
            !parsePciAddr(item.substr(0, plus), p.first) ||
            !parsePciAddr(item.substr(plus + 1), p.second))
            return false;
        out.push_back(p);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return !out.empty();
}

//   STAGE <bus src> <bus dst> <PASS|FAIL|UNTESTED>   host-staged copy test
//   NOKERNEL <bus> <bus>       the gather kernel has no image for that GPU
struct ChildLine {
    enum Kind {
        kInvalid,
        kBegin,
        kDir,
        kEnd,
        kStage,
        kNoKernel
    } kind = kInvalid;
    PciAddr a, b;
    DirVerdict verdict = DirVerdict::kUntested;
};

inline ChildLine parseChildLine(const std::string &line) {
    ChildLine out;
    std::vector<std::string> f;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && line[i] == ' ') ++i;
        size_t j = line.find(' ', i);
        if (j == std::string::npos) j = line.size();
        if (j > i) f.push_back(line.substr(i, j - i));
        i = j;
    }
    if (f.size() < 3 || !parsePciAddr(f[1], out.a) ||
        !parsePciAddr(f[2], out.b))
        return ChildLine();
    if (f[0] == "BEGIN" && f.size() == 3) {
        out.kind = ChildLine::kBegin;
    } else if (f[0] == "END" && f.size() == 3) {
        out.kind = ChildLine::kEnd;
    } else if (f[0] == "NOKERNEL" && f.size() == 3) {
        out.kind = ChildLine::kNoKernel;
    } else if (f[0] == "DIR" && f.size() == 4 &&
               parseDirVerdict(f[3], out.verdict)) {
        out.kind = ChildLine::kDir;
    } else if (f[0] == "STAGE" && f.size() == 4 &&
               parseDirVerdict(f[3], out.verdict) &&
               out.verdict != DirVerdict::kNoPeer) {
        out.kind = ChildLine::kStage;
    } else {
        return ChildLine();
    }
    return out;
}

}  // namespace nvlink_proxy
}  // namespace mooncake

#endif  // MOONCAKE_NVLINK_PROXY_P2P_POLICY_H_
