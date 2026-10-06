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

#ifndef MOONCAKE_NVLINK_PROXY_GATHER_COPY_H_
#define MOONCAKE_NVLINK_PROXY_GATHER_COPY_H_

#include <cuda_runtime.h>

#include <cstdint>

namespace mooncake {
namespace nvlink_proxy {

// One copy of a batch executed by the gather/scatter kernel. Pointers are
// valid in the context of the GPU that runs the kernel (local memory or a
// peer mapping opened in that context).
struct GatherEntry {
    uint64_t dst;
    uint64_t src;
    uint64_t length;
};

// Copies every entry of |entries| (device memory, |count| entries) with one
// kernel launch on |stream|: each thread block copies whole entries using
// the widest access (16/8/4/2/1 bytes) that the relative alignment of source
// and destination allows. |grid_blocks| bounds the number of blocks.
cudaError_t launchGatherCopy(const GatherEntry *entries, uint32_t count,
                             int grid_blocks, cudaStream_t stream);

}  // namespace nvlink_proxy
}  // namespace mooncake

#endif  // MOONCAKE_NVLINK_PROXY_GATHER_COPY_H_
