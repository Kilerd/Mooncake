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

#ifndef MOONCAKE_FRESH_RANDOM_H_
#define MOONCAKE_FRESH_RANDOM_H_

// Random identities that peers must tell apart (engine client ids, request
// id ranges), drawn from the kernel when they are needed. A process restored
// from a checkpoint (CRIU) has the memory of the process it was taken from:
// PRNG state, or any value drawn when that process started, is the same in
// every copy restored from one snapshot, while each getrandom() call returns
// new bytes.

#include <errno.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace mooncake {

inline uint64_t freshRandom64() {
    uint64_t v = 0;
    char *p = reinterpret_cast<char *>(&v);
    size_t got = 0;
#ifdef SYS_getrandom
    while (got < sizeof(v)) {
        const long n = syscall(SYS_getrandom, p + got, sizeof(v) - got, 0);
        if (n > 0) {
            got += static_cast<size_t>(n);
        } else if (n < 0 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
#endif
    if (got < sizeof(v)) {
        // getrandom unavailable (old kernel, seccomp): /dev/urandom.
        const int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
        if (fd >= 0) {
            got = 0;
            while (got < sizeof(v)) {
                const ssize_t n = read(fd, p + got, sizeof(v) - got);
                if (n > 0) {
                    got += static_cast<size_t>(n);
                } else if (n < 0 && errno == EINTR) {
                    continue;
                } else {
                    break;
                }
            }
            close(fd);
        }
    }
    if (got < sizeof(v)) {
        // Last resort, never expected on Linux: the clocks differ between
        // restored copies, the pid may not.
        v ^= static_cast<uint64_t>(
                 std::chrono::steady_clock::now().time_since_epoch().count()) ^
             (static_cast<uint64_t>(
                  std::chrono::system_clock::now().time_since_epoch().count())
              << 17) ^
             (static_cast<uint64_t>(getpid()) << 40);
    }
    return v;
}

}  // namespace mooncake

#endif  // MOONCAKE_FRESH_RANDOM_H_
