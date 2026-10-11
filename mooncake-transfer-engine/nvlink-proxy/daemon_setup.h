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

// Start-up checks of mooncake_nvlink_proxy that need no GPU: the socket and
// its permissions, the peer's credentials, the number of visible GPUs.
//
// Host-only, without CUDA, so that they can be unit-tested without GPUs.

#ifndef MOONCAKE_NVLINK_PROXY_DAEMON_SETUP_H_
#define MOONCAKE_NVLINK_PROXY_DAEMON_SETUP_H_

#include <errno.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace mooncake {
namespace nvlink_proxy {

// Default permissions of the daemon's socket: owner only. Whoever can
// connect can copy between every registered buffer of every engine, so the
// engines must run as the daemon's user (or use --socket-mode 0660 with a
// group they share with it).
constexpr unsigned kDefaultSocketMode = 0600;

// --socket-mode: octal permission bits, at most 0777, that keep read and
// write for the owner (the daemon's own --ping / --stats connect as it), e.g.
// "600", "0660"; "0" or "0060" are refused.
inline bool parseSocketMode(const std::string &s, unsigned *mode) {
    if (s.empty() || s.size() > 4) return false;
    unsigned v = 0;
    for (char c : s) {
        if (c < '0' || c > '7') return false;
        v = v * 8 + unsigned(c - '0');
    }
    if (v > 0777 || (v & 0600) != 0600) return false;
    *mode = v;
    return true;
}

// Creates, binds and listens on a unix stream socket at |path| whose file
// has exactly |mode|. The file never has wider permissions, not even between
// bind() and listen(): it is created under a umask that clears every other
// bit (umask is process-wide: call this before other threads create files).
// Returns the listening fd, or -1 with *err set (errno preserved).
inline int listenUnixSocket(const std::string &path, unsigned mode, int backlog,
                            std::string *err) {
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        *err = "socket path too long";
        errno = ENAMETOOLONG;
        return -1;
    }
    memcpy(addr.sun_path, path.c_str(), path.size() + 1);
    const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        *err = std::string("socket: ") + strerror(errno);
        return -1;
    }
    auto fail = [&](const char *what) {
        const int saved = errno;
        *err = std::string(what) + ": " + strerror(saved);
        close(fd);
        errno = saved;
        return -1;
    };
    const mode_t old = umask(0777 & ~mode_t(mode));
    const int rc = bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
    const int bind_errno = errno;
    umask(old);
    errno = bind_errno;
    if (rc != 0) return fail("bind");
    // Exactly |mode| (the umask can only have removed bits).
    if (chmod(path.c_str(), mode_t(mode)) != 0) {
        const int saved = errno;
        unlink(path.c_str());
        errno = saved;
        return fail("chmod");
    }
    if (listen(fd, backlog) != 0) {
        const int saved = errno;
        unlink(path.c_str());
        errno = saved;
        return fail("listen");
    }
    return fd;
}

// Credentials of the process at the other end of a unix socket (Linux
// SO_PEERCRED). The pid is as seen in the daemon's PID namespace: 0 when the
// peer runs in a PID namespace the daemon cannot see (e.g. another container
// of the pod without a shared process namespace).
struct PeerCred {
    bool known = false;
    int64_t pid = 0;
    uint32_t uid = 0;
    uint32_t gid = 0;
};

inline PeerCred peerCred(int fd) {
    PeerCred out;
#ifdef SO_PEERCRED
    struct ucred cred{};
    socklen_t len = sizeof(cred);
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &len) == 0 &&
        len == sizeof(cred)) {
        out.known = true;
        out.pid = cred.pid;
        out.uid = cred.uid;
        out.gid = cred.gid;
    }
#else
    (void)fd;
#endif
    return out;
}

// A legacy CUDA IPC handle cannot be imported by a process whose pid has the
// same value as the exporter's, even across PID namespaces (the driver takes
// it for the exporting process itself; cudaIpcOpenMemHandle fails). The
// daemon is often PID 1 of its container, as is the first process of every
// other container: an engine that registers memory from its container's
// PID 1 hits this.
inline bool pidValuesCollide(uint32_t client_pid, uint32_t daemon_pid) {
    return client_pid != 0 && client_pid == daemon_pid;
}

// --min-visible-gpus: an error message when fewer than |min_visible| GPUs are
// visible (the container runtime did not expose the node's GPUs), else "".
inline std::string visibleGpuShortfall(size_t visible, int min_visible) {
    if (min_visible <= 0 || visible >= size_t(min_visible)) return "";
    return "only " + std::to_string(visible) +
           " GPU(s) visible to the daemon, --min-visible-gpus " +
           std::to_string(min_visible) +
           " required: the container runtime did not expose the GPUs the "
           "engines use (check NVIDIA_VISIBLE_DEVICES=all on this container "
           "and that the runtime honours it for unprivileged containers, "
           "accept-nvidia-visible-devices-envvar-when-unprivileged); "
           "without them every copy would use the engines' base transport";
}

}  // namespace nvlink_proxy
}  // namespace mooncake

#endif  // MOONCAKE_NVLINK_PROXY_DAEMON_SETUP_H_
