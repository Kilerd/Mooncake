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

#include "gather_copy.h"

namespace mooncake {
namespace nvlink_proxy {
namespace {

constexpr int kThreads = 128;
constexpr int kUnroll = 4;

// Copies [s, s + len) to [d, d + len) with the whole block, given that
// (d - s) is a multiple of sizeof(T): a byte-wise head up to T alignment, a
// body of T-sized accesses (kUnroll independent loads in flight per thread,
// which matters for loads served over NVLink / PCIe), and a byte-wise tail.
template <typename T>
__device__ __forceinline__ void copySpan(char *d, const char *s, uint64_t len) {
    constexpr uint64_t kW = sizeof(T);
    uint64_t head =
        (kW - (reinterpret_cast<uintptr_t>(d) & (kW - 1))) & (kW - 1);
    if (head > len) head = len;
    for (uint64_t i = threadIdx.x; i < head; i += blockDim.x) d[i] = s[i];
    d += head;
    s += head;
    len -= head;
    const uint64_t n = len / kW;
    T *dv = reinterpret_cast<T *>(d);
    const T *sv = reinterpret_cast<const T *>(s);
    const uint64_t step = uint64_t(blockDim.x) * kUnroll;
    uint64_t base = 0;
    for (; base + step <= n; base += step) {
        T v[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u)
            v[u] = sv[base + threadIdx.x + uint64_t(u) * blockDim.x];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u)
            dv[base + threadIdx.x + uint64_t(u) * blockDim.x] = v[u];
    }
    for (uint64_t i = base + threadIdx.x; i < n; i += blockDim.x) dv[i] = sv[i];
    const uint64_t done = n * kW;
    for (uint64_t i = done + threadIdx.x; i < len; i += blockDim.x) d[i] = s[i];
}

__global__ void __launch_bounds__(kThreads)
    gatherCopyKernel(const GatherEntry *__restrict__ entries, uint32_t count) {
    for (uint32_t e = blockIdx.x; e < count; e += gridDim.x) {
        const GatherEntry ent = entries[e];
        char *d = reinterpret_cast<char *>(ent.dst);
        const char *s = reinterpret_cast<const char *>(ent.src);
        const uintptr_t rel = (ent.dst ^ ent.src);
        // The widest access size for which source and destination can be
        // aligned at the same time.
        if ((rel & 15) == 0)
            copySpan<int4>(d, s, ent.length);
        else if ((rel & 7) == 0)
            copySpan<uint2>(d, s, ent.length);
        else if ((rel & 3) == 0)
            copySpan<uint32_t>(d, s, ent.length);
        else if ((rel & 1) == 0)
            copySpan<uint16_t>(d, s, ent.length);
        else
            copySpan<uint8_t>(d, s, ent.length);
    }
}

}  // namespace

cudaError_t launchGatherCopy(const GatherEntry *entries, uint32_t count,
                             int grid_blocks, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    unsigned int grid = count < static_cast<uint32_t>(grid_blocks)
                            ? count
                            : static_cast<unsigned int>(grid_blocks);
    if (grid == 0) grid = 1;
    gatherCopyKernel<<<grid, kThreads, 0, stream>>>(entries, count);
    return cudaGetLastError();
}

}  // namespace nvlink_proxy
}  // namespace mooncake
