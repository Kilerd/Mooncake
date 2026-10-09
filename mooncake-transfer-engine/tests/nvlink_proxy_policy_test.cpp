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

// Host-only parts of the mooncake_nvlink_proxy P2P policy: option parsing,
// the deny matrix, the self-test copy plans and the verifier (no GPU).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "p2p_policy.h"

using namespace mooncake::nvlink_proxy;

TEST(NvlinkProxyPolicy, SelfTestMode) {
    SelfTestMode m = SelfTestMode::kOff;
    EXPECT_TRUE(parseSelfTestMode("enforce", m));
    EXPECT_EQ(m, SelfTestMode::kEnforce);
    EXPECT_TRUE(parseSelfTestMode("warn", m));
    EXPECT_EQ(m, SelfTestMode::kWarn);
    EXPECT_TRUE(parseSelfTestMode("off", m));
    EXPECT_EQ(m, SelfTestMode::kOff);
    EXPECT_FALSE(parseSelfTestMode("Enforce", m));
    EXPECT_FALSE(parseSelfTestMode("", m));
    EXPECT_EQ(m, SelfTestMode::kOff);  // untouched on failure
    EXPECT_STREQ(selfTestModeName(SelfTestMode::kEnforce), "enforce");
}

TEST(NvlinkProxyPolicy, PciAddrForms) {
    PciAddr a;
    ASSERT_TRUE(parsePciAddr("00000000:3b:00.0", a));  // nvidia-smi
    EXPECT_EQ(a.domain, 0u);
    EXPECT_EQ(a.bus, 0x3bu);
    EXPECT_EQ(a.device, 0u);
    PciAddr b;
    ASSERT_TRUE(parsePciAddr("0000:3b:00", b));  // daemon log
    EXPECT_TRUE(a == b);
    ASSERT_TRUE(parsePciAddr("3b:00.0", b));  // lspci
    EXPECT_TRUE(a == b);
    ASSERT_TRUE(parsePciAddr(" 3b:00 ", b));
    EXPECT_TRUE(a == b);
    ASSERT_TRUE(parsePciAddr("0001:C2:1f.7", b));
    EXPECT_EQ(b.domain, 1u);
    EXPECT_EQ(b.bus, 0xc2u);
    EXPECT_EQ(b.device, 0x1fu);
    EXPECT_EQ(formatPciAddr(b), "0001:c2:1f");
    EXPECT_EQ(formatPciAddrShort(b), "0001:c2:1f");
    EXPECT_EQ(formatPciAddrShort(a), "3b:00");
    EXPECT_EQ(formatPciAddr(a), "0000:3b:00");

    for (const char *bad :
         {"", "01", "1:2:3:4", "zz:00", "01:20", "01:00.8", "001:00", "01:000",
          "123456789:01:00", "01:00.", ":01:00", "01:-1"}) {
        EXPECT_FALSE(parsePciAddr(bad, b)) << bad;
    }
}

TEST(NvlinkProxyPolicy, DenyMatrix) {
    PeerPolicy p(4);
    EXPECT_EQ(p.deniedCount(), 0u);
    EXPECT_FALSE(p.denied(0, 1));
    p.deny(0, 1, PeerPolicy::kSelfTestFailed);
    EXPECT_TRUE(p.denied(0, 1));
    EXPECT_FALSE(p.denied(1, 0));  // ordered
    p.denyPair(2, 3, PeerPolicy::kStatic);
    EXPECT_TRUE(p.denied(2, 3));
    EXPECT_TRUE(p.denied(3, 2));
    EXPECT_EQ(p.reason(3, 2), PeerPolicy::kStatic);
    // The first reason sticks.
    p.deny(0, 1, PeerPolicy::kStatic);
    EXPECT_EQ(p.reason(0, 1), PeerPolicy::kSelfTestFailed);
    // A copy within one GPU is never denied; out of range is not either.
    p.denyDevice(0, PeerPolicy::kUntested);
    EXPECT_FALSE(p.denied(0, 0));
    EXPECT_FALSE(p.denied(0, 9));
    EXPECT_FALSE(p.denied(-1, 0));
    for (int o = 1; o < 4; ++o) {
        EXPECT_TRUE(p.denied(0, o));
        EXPECT_TRUE(p.denied(o, 0));
    }
    EXPECT_EQ(p.deniedCount(), 8u);  // 0<->1,2,3 and 2<->3
    EXPECT_EQ(p.deniedList({"a", "b", "c", "d"}),
              "a>b,a>c,a>d,b>a,c>a,c>d,d>a,d>c");
    EXPECT_EQ(PeerPolicy(2).deniedList(), "");
    PeerPolicy q(2);
    q.denyPair(0, 1, PeerPolicy::kStatic);
    EXPECT_EQ(q.deniedList(), "0>1,1>0");
    q.reset(3);
    EXPECT_EQ(q.deniedCount(), 0u);
    EXPECT_EQ(q.numDevices(), 3);
}

namespace {
void checkPlan(const std::vector<TestCopy> &plan, uint64_t buf, bool large,
               uint64_t threshold) {
    ASSERT_FALSE(plan.empty());
    std::vector<std::pair<uint64_t, uint64_t>> dst;
    for (const auto &c : plan) {
        EXPECT_GT(c.len, 0u);
        EXPECT_LE(c.src_off + c.len, buf);
        EXPECT_LE(c.dst_off + c.len, buf);
        if (!large) {
            EXPECT_LT(c.len, threshold);
        }
        dst.push_back({c.dst_off, c.dst_off + c.len});
    }
    std::sort(dst.begin(), dst.end());
    EXPECT_GT(dst.front().first, 0u);  // a guard before the first range
    for (size_t i = 1; i < dst.size(); ++i)
        EXPECT_GT(dst[i].first, dst[i - 1].second)
            << "destination ranges must not touch";
    EXPECT_LT(dst.back().second, buf);  // and after the last one
    if (large) {
        // The first copy-engine entry reaches the gather threshold.
        EXPECT_GE(plan.front().len, std::max<uint64_t>(threshold, 1));
    }
}
}  // namespace

TEST(NvlinkProxyPolicy, SelfTestPlans) {
    const uint64_t thr = 128 * 1024;
    for (uint64_t buf : {1ull << 20, 2ull << 20, 4ull << 20}) {
        for (uint32_t iter = 0; iter < 4; ++iter) {
            SCOPED_TRACE(testing::Message() << buf << " " << iter);
            auto ce = selfTestPlan(buf, iter, true, thr);
            checkPlan(ce, buf, true, thr);
            auto k = selfTestPlan(buf, iter, false, thr);
            checkPlan(k, buf, false, thr);
            // The kernel plan has single 2 KiB rows and misaligned copies.
            EXPECT_GE(std::count_if(k.begin(), k.end(),
                                    [](const TestCopy &c) {
                                        return c.len == kSelfTestRowBytes;
                                    }),
                      8);
            EXPECT_TRUE(std::any_of(k.begin(), k.end(), [](const TestCopy &c) {
                return c.src_off % 8 || c.dst_off % 8 || c.len % 8;
            }));
            EXPECT_TRUE(
                std::any_of(ce.begin(), ce.end(), [](const TestCopy &c) {
                    return c.src_off % 16 || c.dst_off % 16 || c.len % 16;
                }));
        }
    }
    // The 4 MiB plan carries a multi-MiB copy-engine run.
    auto big = selfTestPlan(4ull << 20, 0, true, thr);
    EXPECT_TRUE(std::any_of(big.begin(), big.end(), [](const TestCopy &c) {
        return c.len >= (2ull << 20);
    }));
    // Iterations move the layout.
    auto p0 = selfTestPlan(4ull << 20, 0, true, thr);
    auto p1 = selfTestPlan(4ull << 20, 1, true, thr);
    ASSERT_EQ(p0.size(), p1.size());
    EXPECT_NE(p0[0].dst_off, p1[0].dst_off);
    // Copy-engine plans also carry entries below the threshold (what the
    // copy engines get when the kernel is unusable).
    EXPECT_TRUE(std::any_of(big.begin(), big.end(),
                            [&](const TestCopy &c) { return c.len < thr; }));
    // Whatever the threshold, a buffer of selfTestMinBufferBytes() holds an
    // entry of at least the threshold in every iteration.
    for (uint64_t t : {uint64_t(0), uint64_t(1), uint64_t(128 * 1024),
                       uint64_t(3 << 20), uint64_t(9 << 20) + 5}) {
        const uint64_t buf = selfTestMinBufferBytes(t);
        EXPECT_GE(buf, 1u << 20);
        EXPECT_EQ(buf % (64 * 1024), 0u);
        for (uint32_t iter = 0; iter < 4; ++iter) {
            SCOPED_TRACE(testing::Message()
                         << "threshold " << t << " " << iter);
            auto p = selfTestPlan(buf, iter, true, t);
            ASSERT_FALSE(p.empty());
            EXPECT_GE(p.front().len, t);
            checkPlan(p, buf, true, t);
        }
    }
    // A threshold of 0 (copy engines only) leaves nothing for the kernel and
    // gives the copy engines small entries too.
    EXPECT_TRUE(selfTestPlan(4ull << 20, 0, false, 0).empty());
    auto all_ce = selfTestPlan(4ull << 20, 0, true, 0);
    EXPECT_TRUE(
        std::any_of(all_ce.begin(), all_ce.end(),
                    [](const TestCopy &c) { return c.len < 128 * 1024; }));
}

TEST(NvlinkProxyPolicy, VerifierCatchesCorruptDroppedAndStrayWrites) {
    const uint64_t n = 1ull << 20;
    std::vector<uint8_t> src(n), poison(n), expect(n), got(n);
    fillPattern(src.data(), n, 1);
    fillPattern(poison.data(), n, ~uint64_t(1));
    EXPECT_NE(src, poison);
    const auto plan = selfTestPlan(n, 0, true, 128 * 1024);
    expectedImage(src.data(), poison.data(), n, plan, expect.data());

    // A correct copy.
    got = poison;
    for (const auto &c : plan)
        std::copy(src.begin() + c.src_off, src.begin() + c.src_off + c.len,
                  got.begin() + c.dst_off);
    EXPECT_FALSE(compareImages(expect.data(), got.data(), n).any());

    // One flipped byte.
    auto flipped = got;
    flipped[plan[1].dst_off + 77] ^= 0x40;
    Mismatch m = compareImages(expect.data(), flipped.data(), n);
    EXPECT_EQ(m.bytes, 1u);
    EXPECT_EQ(m.first, plan[1].dst_off + 77);

    // Dropped writes: the destination keeps the poison.
    m = compareImages(expect.data(), poison.data(), n);
    EXPECT_TRUE(m.any());
    EXPECT_EQ(m.first, plan[0].dst_off);

    // A stray write into a guard.
    auto stray = got;
    stray[plan[0].dst_off + plan[0].len] ^= 1;
    EXPECT_EQ(compareImages(expect.data(), stray.data(), n).bytes, 1u);

    // Patterns depend on the seed and are deterministic; lengths that are
    // not a multiple of 8 are filled completely.
    std::vector<uint8_t> a(13, 0), b(13, 0);
    fillPattern(a.data(), a.size(), 7);
    fillPattern(b.data(), b.size(), 7);
    EXPECT_EQ(a, b);
    fillPattern(b.data(), b.size(), 8);
    EXPECT_NE(a, b);
    EXPECT_TRUE(
        std::any_of(a.begin() + 8, a.end(), [](uint8_t v) { return v != 0; }));
}

TEST(NvlinkProxyPolicy, SetOverridesAllButCommandLineDenials) {
    PeerPolicy p(3);
    p.setPair(0, 1, PeerPolicy::kUntested);  // requested, not verified yet
    EXPECT_TRUE(p.denied(0, 1) && p.denied(1, 0));
    p.setPair(0, 1, PeerPolicy::kAllowed);  // the test passed
    EXPECT_FALSE(p.denied(0, 1) || p.denied(1, 0));
    p.setPair(0, 1, PeerPolicy::kSelfTestFailed);
    EXPECT_EQ(p.reason(1, 0), PeerPolicy::kSelfTestFailed);
    p.denyPair(1, 2, PeerPolicy::kStatic);
    p.setPair(1, 2, PeerPolicy::kAllowed);  // never lifts --deny-*
    EXPECT_EQ(p.reason(1, 2), PeerPolicy::kStatic);
    EXPECT_EQ(p.reason(2, 1), PeerPolicy::kStatic);
}

TEST(NvlinkProxyPolicy, Verdicts) {
    using V = DirVerdict;
    for (V v : {V::kPass, V::kFail, V::kUntested, V::kNoPeer}) {
        V back = V::kPass;
        ASSERT_TRUE(parseDirVerdict(dirVerdictName(v), back));
        EXPECT_EQ(back, v);
    }
    V x;
    EXPECT_FALSE(parseDirVerdict("pass", x));
    EXPECT_EQ(combineVerdicts(V::kPass, V::kPass), PairVerdict::kPassed);
    // No peer access in one direction is not judged.
    EXPECT_EQ(combineVerdicts(V::kPass, V::kNoPeer), PairVerdict::kPassed);
    EXPECT_EQ(combineVerdicts(V::kNoPeer, V::kNoPeer), PairVerdict::kPassed);
    // Only data evidence fails a pair.
    EXPECT_EQ(combineVerdicts(V::kUntested, V::kPass), PairVerdict::kUntested);
    EXPECT_EQ(combineVerdicts(V::kNoPeer, V::kUntested),
              PairVerdict::kUntested);
    EXPECT_EQ(combineVerdicts(V::kUntested, V::kFail), PairVerdict::kFailed);
    EXPECT_EQ(combineVerdicts(V::kFail, V::kNoPeer), PairVerdict::kFailed);
}

TEST(NvlinkProxyPolicy, ChildWire) {
    PciAddr a, b, c;
    ASSERT_TRUE(parsePciAddr("0000:3b:00", a));
    ASSERT_TRUE(parsePciAddr("0000:5e:00", b));
    ASSERT_TRUE(parsePciAddr("0001:d8:00", c));
    const std::vector<PciPair> pairs = {{a, b}, {b, c}};
    const std::string arg = encodePciPairs(pairs);
    EXPECT_EQ(arg, "0000:3b:00+0000:5e:00,0000:5e:00+0001:d8:00");
    std::vector<PciPair> back;
    ASSERT_TRUE(parsePciPairs(arg, back));
    ASSERT_EQ(back.size(), 2u);
    EXPECT_TRUE(back[1].first == b && back[1].second == c);
    for (const char *bad :
         {"", "0000:3b:00", "0000:3b:00+", "a+b", "0000:3b:00+0000:5e:00,"}) {
        EXPECT_FALSE(parsePciPairs(bad, back)) << bad;
    }

    ChildLine l = parseChildLine("BEGIN 0000:3b:00 0000:5e:00");
    EXPECT_EQ(l.kind, ChildLine::kBegin);
    EXPECT_TRUE(l.a == a && l.b == b);
    l = parseChildLine("DIR 0000:5e:00 0000:3b:00 FAIL");
    EXPECT_EQ(l.kind, ChildLine::kDir);
    EXPECT_TRUE(l.a == b && l.b == a);
    EXPECT_EQ(l.verdict, DirVerdict::kFail);
    EXPECT_EQ(parseChildLine("END 0000:3b:00 0000:5e:00").kind,
              ChildLine::kEnd);
    for (const char *bad :
         {"", "BEGIN", "BEGIN 0000:3b:00", "DIR 0000:3b:00 0000:5e:00",
          "DIR 0000:3b:00 0000:5e:00 MAYBE", "END x y", "HELLO 3b:00 5e:00",
          "BEGIN 0000:3b:00 0000:5e:00 extra"}) {
        EXPECT_EQ(parseChildLine(bad).kind, ChildLine::kInvalid) << bad;
    }
}
