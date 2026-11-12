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

// How the nvlink_proxy transport registers device memory with the daemon,
// without GPUs: this file defines the CUDA runtime and driver calls that
// registerLocalMemory makes, and they take precedence over libcudart and
// libcuda for the engine linked into this test.
//
// The stand-in has two GPUs and per-thread context stacks. "Device memory"
// is host memory listed as a cudaMalloc block or a virtual memory
// management allocation of one GPU. As on the 5090s, cuMemGetAddressRange
// finds an allocation only while a context of its GPU is current
// (CUDA_ERROR_NOT_FOUND otherwise).

#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "transfer_engine.h"
#include "transport/nvlink_proxy_transport/nvlink_proxy_protocol.h"

using namespace mooncake;
using namespace mooncake::nvlink_proxy;

namespace {

constexpr int kGpus = 2;

enum class Kind { kCudaMalloc, kVmm, kVmmWithoutFd };

struct Region {
    uintptr_t base = 0;
    size_t size = 0;
    int dev = 0;
    Kind kind = Kind::kCudaMalloc;
    uint64_t id = 0;  // CU_POINTER_ATTRIBUTE_BUFFER_ID, allocation handle
};

std::mutex g_mu;
std::vector<Region> g_regions;
std::atomic<int> g_ipc_exports{0};  // cudaIpcGetMemHandle calls
std::atomic<int> g_fd_exports{0};   // cuMemExportToShareableHandle calls
std::atomic<int> g_primary_refs{0};
thread_local std::vector<int> t_ctx;  // context stack (GPU of each)

CUcontext ctxOf(int dev) {
    return reinterpret_cast<CUcontext>(static_cast<uintptr_t>(0x1000 + dev));
}

int devOfCtx(CUcontext c) {
    return static_cast<int>(reinterpret_cast<uintptr_t>(c) - 0x1000);
}

bool find(uintptr_t p, Region *out) {
    std::lock_guard<std::mutex> lock(g_mu);
    for (auto &r : g_regions) {
        if (p >= r.base && p < r.base + r.size) {
            *out = r;
            return true;
        }
    }
    return false;
}

bool findId(uint64_t id, Region *out) {
    std::lock_guard<std::mutex> lock(g_mu);
    for (auto &r : g_regions) {
        if (r.id == id) {
            *out = r;
            return true;
        }
    }
    return false;
}

// Memory of |size| bytes listed as device memory of GPU |dev|.
struct DeviceMemory {
    DeviceMemory(int dev, Kind kind, size_t size) : mem(size) {
        std::lock_guard<std::mutex> lock(g_mu);
        region.base = reinterpret_cast<uintptr_t>(mem.data());
        region.size = size;
        region.dev = dev;
        region.kind = kind;
        region.id = 100 + g_regions.size();
        g_regions.push_back(region);
    }
    ~DeviceMemory() {
        std::lock_guard<std::mutex> lock(g_mu);
        for (auto it = g_regions.begin(); it != g_regions.end(); ++it) {
            if (it->base == region.base) {
                g_regions.erase(it);
                break;
            }
        }
    }
    void *at(size_t off) { return mem.data() + off; }
    std::vector<char> mem;
    Region region;
};

}  // namespace

extern "C" {

cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attr,
                                     const void *p) {
    memset(attr, 0, sizeof(*attr));
    Region r;
    if (!find(reinterpret_cast<uintptr_t>(p), &r)) {
        attr->type = cudaMemoryTypeUnregistered;
        attr->device = -2;
        return cudaSuccess;
    }
    attr->type = cudaMemoryTypeDevice;
    attr->device = r.dev;
    attr->devicePointer = const_cast<void *>(p);
    return cudaSuccess;
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *p) {
    g_ipc_exports++;
    Region r;
    if (!find(reinterpret_cast<uintptr_t>(p), &r) ||
        r.kind != Kind::kCudaMalloc || r.base != reinterpret_cast<uintptr_t>(p))
        return cudaErrorInvalidValue;
    memset(handle, 0, sizeof(*handle));
    memcpy(handle->reserved, &r.id, sizeof(r.id));
    return cudaSuccess;
}

static cudaError_t fillProps(cudaDeviceProp *p, int d) {
    if (d < 0 || d >= kGpus) return cudaErrorInvalidDevice;
    memset(p, 0, sizeof(*p));
    snprintf(p->name, sizeof(p->name), "Stand-in GPU");
    for (int i = 0; i < 16; ++i)
        p->uuid.bytes[i] = static_cast<char>(d * 16 + i);
    return cudaSuccess;
}
#if CUDART_VERSION >= 12000
cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp *p, int d) {
    return fillProps(p, d);
}
#endif
#if CUDART_VERSION >= 13000 || !defined(cudaGetDeviceProperties)
#undef cudaGetDeviceProperties
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int d) {
    return fillProps(p, d);
}
#endif

cudaError_t cudaSetDevice(int d) {
    if (d < 0 || d >= kGpus) return cudaErrorInvalidDevice;
    if (t_ctx.empty())
        t_ctx.push_back(d);
    else
        t_ctx.back() = d;
    return cudaSuccess;
}

cudaError_t cudaGetLastError(void) { return cudaSuccess; }

CUresult cuDeviceGet(CUdevice *d, int ordinal) {
    if (ordinal < 0 || ordinal >= kGpus) return CUDA_ERROR_INVALID_DEVICE;
    *d = ordinal;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext *c) {
    *c = t_ctx.empty() ? nullptr : ctxOf(t_ctx.back());
    return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice(CUdevice *d) {
    if (t_ctx.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    *d = t_ctx.back();
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *c, CUdevice d) {
    if (d < 0 || d >= kGpus) return CUDA_ERROR_INVALID_DEVICE;
    g_primary_refs++;
    *c = ctxOf(d);
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice d) {
    if (d < 0 || d >= kGpus) return CUDA_ERROR_INVALID_DEVICE;
    g_primary_refs--;
    return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent(CUcontext c) {
    t_ctx.push_back(devOfCtx(c));
    return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext *c) {
    if (t_ctx.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    *c = ctxOf(t_ctx.back());
    t_ctx.pop_back();
    return CUDA_SUCCESS;
}

CUresult cuMemGetAddressRange(CUdeviceptr *base, size_t *size, CUdeviceptr p) {
    if (t_ctx.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    Region r;
    if (!find(static_cast<uintptr_t>(p), &r) || r.dev != t_ctx.back())
        return CUDA_ERROR_NOT_FOUND;
    *base = static_cast<CUdeviceptr>(r.base);
    *size = r.size;
    return CUDA_SUCCESS;
}

CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *h, void *p) {
    Region r;
    if (!find(reinterpret_cast<uintptr_t>(p), &r) ||
        r.kind == Kind::kCudaMalloc)
        return CUDA_ERROR_INVALID_VALUE;
    *h = r.id;
    return CUDA_SUCCESS;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle) { return CUDA_SUCCESS; }

CUresult cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp *prop, CUmemGenericAllocationHandle h) {
    Region r;
    if (!findId(h, &r)) return CUDA_ERROR_INVALID_VALUE;
    memset(prop, 0, sizeof(*prop));
    prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop->location.id = r.dev;
    prop->requestedHandleTypes = r.kind == Kind::kVmm
                                     ? CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
                                     : CU_MEM_HANDLE_TYPE_NONE;
    return CUDA_SUCCESS;
}

CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attr,
                               CUdeviceptr p) {
    Region r;
    if (attr != CU_POINTER_ATTRIBUTE_BUFFER_ID ||
        !find(static_cast<uintptr_t>(p), &r))
        return CUDA_ERROR_INVALID_VALUE;
    *static_cast<unsigned long long *>(data) = r.id;
    return CUDA_SUCCESS;
}

CUresult cuMemExportToShareableHandle(void *out, CUmemGenericAllocationHandle h,
                                      CUmemAllocationHandleType type,
                                      unsigned long long) {
    Region r;
    if (type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR || !findId(h, &r) ||
        r.kind != Kind::kVmm)
        return CUDA_ERROR_INVALID_VALUE;
    g_fd_exports++;
    int fds[2];
    if (pipe(fds) != 0) return CUDA_ERROR_OPERATING_SYSTEM;
    close(fds[1]);
    *static_cast<int *>(out) = fds[0];
    return CUDA_SUCCESS;
}

}  // extern "C"

#if defined(USE_NVLINK_PROXY) && defined(USE_TCP)
namespace {

// A stand-in for mooncake_nvlink_proxy that accepts every block and
// records what it was asked to register.
class FakeDaemon {
   public:
    struct Registration {
        MsgType type = MsgType::kRegister;
        uint64_t base = 0, size = 0;
        uint8_t uuid0 = 0;  // first byte of the GPU UUID
        bool fd = false;    // a descriptor came with it (SCM_RIGHTS)
    };

    FakeDaemon() {
        path_ = "/tmp/mc-nvlp-reg-" + std::to_string(getpid()) + ".sock";
        unlink(path_.c_str());
        listen_fd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        memcpy(addr.sun_path, path_.c_str(), path_.size() + 1);
        EXPECT_EQ(
            bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)),
            0);
        EXPECT_EQ(listen(listen_fd_, 16), 0);
        acceptor_ = std::thread([this] { acceptLoop(); });
    }

    ~FakeDaemon() {
        stopping_ = true;
        ::shutdown(listen_fd_, SHUT_RDWR);
        acceptor_.join();
        {
            std::lock_guard<std::mutex> lock(mu_);
            for (int fd : open_) ::shutdown(fd, SHUT_RDWR);
        }
        for (auto &t : serving_) t.join();
        close(listen_fd_);
        unlink(path_.c_str());
    }

    const std::string &path() const { return path_; }

    std::vector<Registration> registrations() {
        std::lock_guard<std::mutex> lock(mu_);
        return registrations_;
    }

    bool waitControl(int timeout_ms) {
        const auto until = std::chrono::steady_clock::now() +
                           std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < until) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (controls_ > 0) return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return false;
    }

   private:
    void acceptLoop() {
        while (!stopping_) {
            const int fd = accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
            if (fd < 0) continue;
            std::lock_guard<std::mutex> lock(mu_);
            open_.insert(fd);
            serving_.emplace_back([this, fd] { serve(fd); });
        }
    }

    static bool readAll(int fd, void *buf, size_t len) {
        char *p = static_cast<char *>(buf);
        while (len > 0) {
            const ssize_t n = recv(fd, p, len, 0);
            if (n <= 0) return false;
            p += n;
            len -= static_cast<size_t>(n);
        }
        return true;
    }

    // The header, and the descriptor passed with it (-1: none).
    static bool readHeader(int fd, MsgHeader &h, int &passed) {
        passed = -1;
        iovec iov{&h, sizeof(h)};
        alignas(cmsghdr) char ctrl[CMSG_SPACE(sizeof(int))];
        msghdr m{};
        m.msg_iov = &iov;
        m.msg_iovlen = 1;
        m.msg_control = ctrl;
        m.msg_controllen = sizeof(ctrl);
        if (recvmsg(fd, &m, MSG_WAITALL) != sizeof(h)) return false;
        cmsghdr *c = CMSG_FIRSTHDR(&m);
        if (c && c->cmsg_type == SCM_RIGHTS)
            memcpy(&passed, CMSG_DATA(c), sizeof(passed));
        return true;
    }

    static void reply(int fd, const MsgHeader &req, const void *payload,
                      size_t len) {
        MsgHeader h = req;
        h.status = static_cast<uint32_t>(ProxyStatus::kOk);
        h.payload_len = static_cast<uint32_t>(len);
        std::string out(reinterpret_cast<const char *>(&h), sizeof(h));
        if (len) out.append(static_cast<const char *>(payload), len);
        (void)send(fd, out.data(), out.size(), MSG_NOSIGNAL);
    }

    void serve(int fd) {
        MsgHeader h;
        int passed;
        std::string payload;
        while (readHeader(fd, h, passed)) {
            payload.resize(h.payload_len);
            if (h.payload_len && !readAll(fd, &payload[0], h.payload_len))
                break;
            const auto type = static_cast<MsgType>(h.type);
            switch (type) {
                case MsgType::kHello: {
                    HelloReq req;
                    memcpy(&req, payload.data(), sizeof(req));
                    if (req.flags & kHelloFlagControl) {
                        std::lock_guard<std::mutex> lock(mu_);
                        ++controls_;
                    }
                    HelloResp resp{};
                    resp.daemon_epoch = 0x5eed;
                    resp.num_devices = kGpus;
                    reply(fd, h, &resp, sizeof(resp));
                    break;
                }
                case MsgType::kPing: {
                    PingResp resp{};
                    resp.daemon_epoch = 0x5eed;
                    reply(fd, h, &resp, sizeof(resp));
                    break;
                }
                case MsgType::kRegister:
                case MsgType::kRegisterFd: {
                    RegisterReq req;
                    memcpy(&req, payload.data(), sizeof(req));
                    Registration rec;
                    rec.type = type;
                    rec.base = req.base;
                    rec.size = req.size;
                    rec.uuid0 = req.gpu_uuid[0];
                    rec.fd = passed >= 0;
                    {
                        std::lock_guard<std::mutex> lock(mu_);
                        registrations_.push_back(rec);
                    }
                    reply(fd, h, nullptr, 0);
                    break;
                }
                case MsgType::kRelease: {
                    ReleaseResp resp{};
                    reply(fd, h, &resp, sizeof(resp));
                    break;
                }
                default:
                    reply(fd, h, nullptr, 0);
            }
            if (passed >= 0) close(passed);
        }
        std::lock_guard<std::mutex> lock(mu_);
        close(fd);
        open_.erase(fd);
    }

    std::string path_;
    int listen_fd_ = -1;
    std::atomic<bool> stopping_{false};
    std::thread acceptor_;
    std::mutex mu_;
    std::vector<std::thread> serving_;
    std::set<int> open_;
    int controls_ = 0;
    std::vector<Registration> registrations_;
};

// Engine variables of one test, unset again at its end.
struct Env {
    explicit Env(std::vector<std::pair<const char *, const char *>> vars)
        : vars_(std::move(vars)) {
        for (auto &kv : vars_) setenv(kv.first, kv.second, 1);
    }
    ~Env() {
        for (auto &kv : vars_) unsetenv(kv.first);
        unsetenv("MC_NVLINK_PROXY_SOCKET");
        unsetenv("MC_FORCE_TCP");
    }
    std::vector<std::pair<const char *, const char *>> vars_;
};

// A TCP engine with nvlink_proxy installed against |daemon|.
std::unique_ptr<TransferEngine> proxyEngine(FakeDaemon &daemon) {
    setenv("MC_FORCE_TCP", "1", 1);
    setenv("MC_NVLINK_PROXY_SOCKET", daemon.path().c_str(), 1);
    auto engine = std::make_unique<TransferEngine>(true);
    EXPECT_EQ(engine->init(P2PHANDSHAKE, "127.0.0.1", "", 0), 0);
    EXPECT_NE(engine->getTransport("nvlink_proxy"), nullptr);
    EXPECT_TRUE(daemon.waitControl(5000));
    return engine;
}

constexpr size_t kMiB = 1 << 20;

}  // namespace

// The test thread works on GPU 0 and registers buffers of GPU 1: each is
// looked up and exported with GPU 1's context current (the daemon gets
// the allocation's base, size and GPU), and GPU 0's context is current
// again afterwards.
TEST(NvlinkProxyRegister, BufferOfAnotherGpuUsesThatGpusContext) {
    Env env({{"MC_NVLINK_PROXY_VMM", "1"}});
    FakeDaemon daemon;
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    DeviceMemory legacy(1, Kind::kCudaMalloc, 4 * kMiB);
    DeviceMemory vmm(1, Kind::kVmm, 4 * kMiB);
    DeviceMemory own(0, Kind::kCudaMalloc, 4 * kMiB);
    auto engine = proxyEngine(daemon);
    const int refs = g_primary_refs.load();

    EXPECT_EQ(engine->registerLocalMemory(legacy.at(kMiB), kMiB, "*"), 0);
    EXPECT_EQ(engine->registerLocalMemory(vmm.at(0), 2 * kMiB, "*"), 0);
    EXPECT_EQ(engine->registerLocalMemory(own.at(0), kMiB, "*"), 0);

    const auto regs = daemon.registrations();
    ASSERT_EQ(regs.size(), 3u);
    EXPECT_EQ(regs[0].type, MsgType::kRegister);
    EXPECT_EQ(regs[0].base, legacy.region.base);
    EXPECT_EQ(regs[0].size, legacy.region.size);
    EXPECT_EQ(regs[0].uuid0, 16);  // GPU 1
    EXPECT_EQ(regs[1].type, MsgType::kRegisterFd);
    EXPECT_EQ(regs[1].base, vmm.region.base);
    EXPECT_EQ(regs[1].uuid0, 16);
    EXPECT_TRUE(regs[1].fd);
    EXPECT_EQ(regs[2].type, MsgType::kRegister);
    EXPECT_EQ(regs[2].uuid0, 0);  // GPU 0
    ASSERT_EQ(t_ctx.size(), 1u);
    EXPECT_EQ(t_ctx.back(), 0);
    EXPECT_EQ(g_primary_refs.load(), refs);

    EXPECT_EQ(engine->unregisterLocalMemory(legacy.at(kMiB)), 0);
    EXPECT_EQ(engine->unregisterLocalMemory(vmm.at(0)), 0);
    EXPECT_EQ(engine->unregisterLocalMemory(own.at(0)), 0);
    EXPECT_EQ(engine->shutdown(), 0);
    t_ctx.clear();
}

// Without a current context at all (a thread that never used CUDA).
TEST(NvlinkProxyRegister, ThreadWithoutContext) {
    Env env({{"MC_NVLINK_PROXY_VMM", "1"}});
    FakeDaemon daemon;
    DeviceMemory vmm(1, Kind::kVmm, 4 * kMiB);
    auto engine = proxyEngine(daemon);
    t_ctx.clear();
    EXPECT_EQ(engine->registerLocalMemory(vmm.at(0), kMiB, "*"), 0);
    const auto regs = daemon.registrations();
    ASSERT_EQ(regs.size(), 1u);
    EXPECT_EQ(regs[0].type, MsgType::kRegisterFd);
    EXPECT_TRUE(t_ctx.empty());
    EXPECT_EQ(engine->unregisterLocalMemory(vmm.at(0)), 0);
    EXPECT_EQ(engine->shutdown(), 0);
}

// cudaMalloc memory is exported with cudaIpcGetMemHandle.
TEST(NvlinkProxyRegister, CudaMallocMemoryUsesLegacyIpc) {
    Env env({});
    FakeDaemon daemon;
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    DeviceMemory legacy(0, Kind::kCudaMalloc, 4 * kMiB);
    auto engine = proxyEngine(daemon);
    g_ipc_exports = 0;
    EXPECT_EQ(engine->registerLocalMemory(legacy.at(0), kMiB, "*"), 0);
    EXPECT_EQ(g_ipc_exports.load(), 1);
    const auto regs = daemon.registrations();
    ASSERT_EQ(regs.size(), 1u);
    EXPECT_EQ(regs[0].type, MsgType::kRegister);
    EXPECT_EQ(engine->unregisterLocalMemory(legacy.at(0)), 0);
    EXPECT_EQ(engine->shutdown(), 0);
    t_ctx.clear();
}

#endif
