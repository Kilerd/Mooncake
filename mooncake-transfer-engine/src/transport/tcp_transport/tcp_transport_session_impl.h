// Copyright 2024 KVCache.AI
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

// Private implementation fragment included by tcp_transport.cpp from inside
// namespace mooncake. Keeping the protocol/session state here makes the public
// transport entry points and the lane scheduler independently reviewable
// without changing linkage or lifetime behavior.

static size_t getChunkSize() {
    static const size_t val = [] {
        const char* env = std::getenv("MC_TCP_SLICE_SIZE");
        if (env) {
            try {
                size_t v = std::stoull(env);
                if (v > 0) return v;
                LOG(WARNING)
                    << "Ignore non-positive MC_TCP_SLICE_SIZE value: " << env
                    << ", using default 65536";
            } catch (const std::exception& e) {
                // A non-numeric or out-of-range value makes std::stoull throw;
                // fall through to the default instead of letting the exception
                // propagate out of this static initializer and abort the
                // transfer that first reads the chunk size.
                LOG(WARNING)
                    << "Invalid MC_TCP_SLICE_SIZE value: " << env
                    << ". Error: " << e.what() << ", using default 65536";
            }
        }
        return size_t(65536);  // 64KB default
    }();
    return val;
}

struct SessionHeader {
    uint64_t size;
    uint64_t addr;
    uint8_t opcode;
};

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
// Returns the CUDA device ordinal if addr is device memory, or -1 otherwise.
// Callers must call cudaSetDevice before any cudaMemcpy to avoid implicit
// GPU 0 context creation.
static int getCudaDeviceId(void* addr) {
    cudaPointerAttributes attributes;
    auto status = cudaPointerGetAttributes(&attributes, addr);
    if (status != cudaSuccess) return -1;
    if (attributes.type == cudaMemoryTypeDevice) return attributes.device;
    return -1;
}

#ifdef USE_MACA
static cudaError_t copyTcpCudaMemory(void* dst, const void* src, size_t size) {
    cudaStream_t stream;
    cudaError_t status =
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) return status;

    status = cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(stream);
    }

    cudaError_t destroy_status = cudaStreamDestroy(stream);
    return status == cudaSuccess ? destroy_status : status;
}
#endif
#endif

// Forward declaration
class TcpTransport;

using ValidateAddrFn = std::function<bool(uint64_t, uint64_t)>;

// --- Acknowledged framing (protocol v2, #2086) ------------------------------
// v1 framing gives the initiator no channel to learn whether the receiver
// applied (or even accepted) a WRITE: COMPLETED fires when the final chunk
// enters the initiator's kernel socket buffer, while megabytes may still be
// in flight toward destination memory, and a rejected request is silently
// "successful". v2 requests set the high bit of the opcode; the server then
// (a) prefixes every READ response with an 8-byte status frame and (b) sends
// an 8-byte status frame for WRITE only after the final chunk has been
// applied to destination memory. Initiators enable v2 only when the target
// segment advertises tcp_proto_version >= 2, so old servers never see
// flagged opcodes and old initiators keep receiving v1 framing.
static constexpr uint8_t kOpcodeV2Flag = 0x80;
// Status frames carry a magic in the high 32 bits so that a v2 initiator
// which reaches a v1 server through a stale descriptor (v1 treats unknown
// opcodes as READ and immediately streams payload bytes) fails fast on the
// first frame instead of misinterpreting the stream. Residual risk: payload
// bytes that happen to equal a valid frame (2^-64 per request, data
// dependent) are indistinguishable in-band; eliminating that would need a
// nonce/checksum handshake, which this deliberately avoids.
static constexpr uint64_t kStatusMagic = 0x4D435456ull << 32;  // "MCTV"
static constexpr uint64_t kStatusOk = kStatusMagic | 0;
static constexpr uint64_t kStatusAddrRejected = kStatusMagic | 1;
static inline bool statusFrameValid(uint64_t frame) {
    return (frame & 0xFFFFFFFF00000000ull) == kStatusMagic;
}

// --- Deadlines, request ids and cancellation (protocol v3) ------------------
// A v3 request sets kOpcodeV3Flag next to kOpcodeV2Flag (initiators set it
// only when the target segment advertises tcp_proto_version >= 3, so a v2
// server never sees it) and follows the 24-byte header with
// SessionHeaderV3Ext: a request id chosen by the initiator and the time it
// has left (microseconds, 0 = no deadline). The receiver's deadline is the
// header's arrival plus that time. Status frames of a v3 request:
// - kStatusOkV3: READ status before the data; WRITE ack once every chunk was
//   applied. (A v2 kStatusOk in its place means a server that did not
//   understand the request -- e.g. an older one behind a stale descriptor
//   -- and fails it.)
// - kStatusDeadline / kStatusCanceled: the receiver dropped the request's
//   chunks from the moment its deadline passed / it was canceled, and
//   applies nothing more of it.
// - kStatusAddrRejected: as in v2; nothing was applied.
// Opcode kOpcodeCancel (v3 only, on its own connection) names a request id:
// the receiver makes sure nothing more of that request is applied -- also
// when its header has not arrived yet -- and answers with the request's
// final status (kStatusOkV3 if it had been applied in full) followed by the
// request id (CancelReply, 16 bytes; a server that does not know v3 cannot
// produce it). That answer is the initiator's proof that a cut WRITE is
// quiesced at the receiver.
static constexpr uint8_t kOpcodeV3Flag = 0x40;
static constexpr uint8_t kOpcodeMask = 0x3f;
static constexpr uint8_t kOpcodeCancel = 2;
static constexpr uint64_t kStatusDeadline = kStatusMagic | 2;
static constexpr uint64_t kStatusCanceled = kStatusMagic | 3;
static constexpr uint64_t kStatusOkV3 = kStatusMagic | 4;
// ClientSession's unconfirmed_request for a WRITE cut mid-body on a
// connection that cannot confirm anything (protocol v1/v2): unconfirmed for
// good. (nextTcpRequestId never returns it.)
static constexpr uint64_t kUnprovableRequest = ~0ull;

struct SessionHeaderV3Ext {
    uint64_t request_id;    // little-endian
    uint64_t remaining_us;  // little-endian, 0 = no deadline
};
static_assert(sizeof(SessionHeaderV3Ext) == 16, "v3 header extension");

struct CancelReply {
    uint64_t status;      // little-endian, the request's final status
    uint64_t request_id;  // little-endian, echoed
};
static_assert(sizeof(CancelReply) == 16, "v3 cancel reply");

// The final statuses that prove a v3 request is quiesced at its receiver.
static inline bool v3FinalStatus(uint64_t frame) {
    return frame == kStatusOkV3 || frame == kStatusDeadline ||
           frame == kStatusCanceled || frame == kStatusAddrRejected;
}

static uint64_t nextTcpRequestId() {
    static std::atomic<uint64_t> next{[] {
        std::random_device rd;
        uint64_t v = (uint64_t(rd()) << 32) ^ rd();
        return v ? v : 1;
    }()};
    uint64_t id = next.fetch_add(1, std::memory_order_relaxed);
    while (id == 0 || id == kUnprovableRequest)
        id = next.fetch_add(1, std::memory_order_relaxed);
    return id;
}

static int64_t steadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
// Tests: a server that does not know v3 (reads 0x40 as part of the opcode,
// as servers before v3 did), and a delay before each chunk a server applies.
static bool testServerIgnoresV3() {
    const char* env = std::getenv("MC_TCP_TEST_SERVER_IGNORE_V3");
    return env && env[0] == '1';
}
// Tests: initiators do not cut a transfer at its deadline (the receiver's
// own deadline handling then shows on its own).
static bool testClientNoCut() {
    const char* env = std::getenv("MC_TCP_TEST_CLIENT_NO_CUT");
    return env && env[0] == '1';
}
static void testServerChunkDelay() {
    const char* env = std::getenv("MC_TCP_TEST_SERVER_CHUNK_DELAY_MS");
    if (env && *env) {
        const int ms = std::atoi(env);
        if (ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
}
#endif

struct ServerSession;

// Streams for copying received TCP payload into device memory, one per
// device, created on first use and owned by the transport. They are
// non-blocking: these copies neither wait for nor hold up work the process
// queued on the legacy default stream.
class TcpDeviceStreams {
   public:
#ifdef USE_CUDA
    ~TcpDeviceStreams() {
        for (auto& kv : streams_) cudaStreamDestroy(kv.second);
    }
    // |device|'s stream (the caller made |device| current); nullptr when
    // none could be created.
    cudaStream_t get(int device) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = streams_.find(device);
        if (it != streams_.end()) return it->second;
        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
            cudaSuccess)
            return nullptr;
        streams_.emplace(device, stream);
        return stream;
    }

   private:
    std::mutex mu_;
    std::unordered_map<int, cudaStream_t> streams_;
#endif
};

// v3 requests a server has seen, for cancellation by request id. Finished
// requests (and ids canceled before their header arrived) are remembered
// for a while so a late cancel still gets the final status.
class ServerRequestRegistry {
   public:
    struct Running {
        std::weak_ptr<ServerSession> session;
        asio::any_io_executor executor;
    };

    // A request's header arrived. False when it was canceled before that:
    // the request must be dropped.
    bool begin(uint64_t id, std::weak_ptr<ServerSession> session,
               asio::any_io_executor executor) {
        std::lock_guard<std::mutex> lock(mu_);
        auto done = done_.find(id);
        if (done != done_.end()) return false;
        running_[id] = Running{std::move(session), std::move(executor)};
        return true;
    }

    void finish(uint64_t id, uint64_t status) {
        std::lock_guard<std::mutex> lock(mu_);
        running_.erase(id);
        rememberLocked(id, status);
    }

    enum class Lookup { kRunning, kDone };
    // For a cancel of |id|: the running request, or its final status (an
    // unknown id is recorded as canceled now, so its header is dropped if it
    // still arrives).
    Lookup lookup(uint64_t id, Running* running, uint64_t* status) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = running_.find(id);
        if (it != running_.end()) {
            *running = it->second;
            return Lookup::kRunning;
        }
        auto done = done_.find(id);
        if (done != done_.end()) {
            *status = done->second;
            return Lookup::kDone;
        }
        rememberLocked(id, kStatusCanceled);
        *status = kStatusCanceled;
        return Lookup::kDone;
    }

   private:
    void rememberLocked(uint64_t id, uint64_t status) {
        if (done_.emplace(id, status).second) order_.push_back(id);
        while (order_.size() > kRemembered) {
            done_.erase(order_.front());
            order_.pop_front();
        }
    }

    static constexpr size_t kRemembered = 1 << 16;
    std::mutex mu_;
    std::unordered_map<uint64_t, Running> running_;
    std::unordered_map<uint64_t, uint64_t> done_;
    std::deque<uint64_t> order_;
};

// Operational escape hatch: MC_TCP_PROTO=1 forces initiators to speak the
// legacy unacknowledged framing even to v2-capable servers, MC_TCP_PROTO=2
// the v2 framing without deadlines and request ids even to v3-capable ones.
// Also used by tests to cover the mixed-version matrix in one process.
static int maxTcpProto() {
    // Read per call (startTransfer already does metadata lookups; getenv is
    // noise) so tests can cover every protocol mode in one process.
    const char* env = std::getenv("MC_TCP_PROTO");
    if (env && (env[0] == '1' || env[0] == '2') && env[1] == '\0')
        return env[0] - '0';
    return 3;
}

// Server-side session: handles transfer requests on a persistent connection.
// The session owns the socket; ending the callback chain without rearming
// (start()/next handler) drops the last reference and closes the connection.
struct ServerSession : public std::enable_shared_from_this<ServerSession> {
    explicit ServerSession(
        std::shared_ptr<tcpsocket> socket, ValidateAddrFn validate_addr,
        std::shared_ptr<ServerRequestRegistry> registry = nullptr,
        std::shared_ptr<TcpDeviceStreams> device_streams = nullptr)
        : socket_(std::move(socket)),
          validate_addr_(std::move(validate_addr)),
          registry_(std::move(registry)),
          device_streams_(std::move(device_streams)) {}

    ~ServerSession() {
        // A v3 request this session never finished (connection lost, or
        // cut by a cancel) applies nothing more once the session is gone.
        if (registry_ && request_open_)
            registry_->finish(request_id_, kStatusCanceled);
    }

    std::shared_ptr<tcpsocket> socket_;
    ValidateAddrFn validate_addr_;
    std::shared_ptr<ServerRequestRegistry> registry_;
    std::shared_ptr<TcpDeviceStreams> device_streams_;
    SessionHeader header_;
    uint64_t total_transferred_bytes_;
    char* local_buffer_;
    bool v2_ = false;
    uint64_t status_frame_;
    // v3 request state (all on this session's executor).
    bool v3_ = false;
    SessionHeaderV3Ext ext_{};
    uint64_t request_id_ = 0;
    bool request_open_ = false;  // registered, not finished yet
    std::chrono::steady_clock::time_point deadline_{};
    bool has_deadline_ = false;
    bool canceled_ = false;     // by a cancel of this request
    uint64_t drop_status_ = 0;  // kStatusDeadline/kStatusCanceled once dropping
    std::vector<char> bounce_;
    CancelReply cancel_reply_{};

    void start() {
        total_transferred_bytes_ = 0;
        v3_ = false;
        request_open_ = false;
        has_deadline_ = false;
        canceled_ = false;
        drop_status_ = 0;
        readHeader();
    }

    // A cancel of request |id|: if this session is still running it, nothing
    // more of it is applied from now on (every apply runs on this executor
    // and checks the flag), and the connection is closed so that no read
    // waits for a body the initiator stopped sending. A request this session
    // already finished keeps its final status, and a later request on the
    // same connection is left alone.
    void cancelByPeer(uint64_t id) {
        if (!request_open_ || request_id_ != id) return;
        canceled_ = true;
        request_open_ = false;
        if (registry_) registry_->finish(request_id_, kStatusCanceled);
        if (socket_ && socket_->is_open()) {
            asio::error_code ec;
            socket_->cancel(ec);
            socket_->close(ec);
        }
    }

   private:
    // Send an 8-byte status frame, then run `next` (or end the session —
    // closing the connection — when `next` is empty or the send fails).
    void sendStatus(uint64_t status, std::function<void()> next) {
        status_frame_ = htole64(status);
        auto self(shared_from_this());
        asio::async_write(*socket_,
                          asio::buffer(&status_frame_, sizeof(status_frame_)),
                          [this, self, next = std::move(next)](
                              const asio::error_code& ec, std::size_t) {
                              if (ec)
                                  return;  // connection closes with the session
                              if (next) next();
                          });
    }

    // Answer a cancel request (and end this session).
    void sendCancelReply(uint64_t status) {
        cancel_reply_.status = htole64(status);
        cancel_reply_.request_id = htole64(request_id_);
        auto self(shared_from_this());
        asio::async_write(*socket_,
                          asio::buffer(&cancel_reply_, sizeof(cancel_reply_)),
                          [self](const asio::error_code&, std::size_t) {});
    }

    // The v3 request ended: |status| is final for a later cancel.
    void finishRequest(uint64_t status) {
        if (!request_open_) return;
        request_open_ = false;
        if (registry_) registry_->finish(request_id_, status);
    }

    // True when the chunk at hand must be dropped (v3: canceled, or the
    // receiver's deadline has passed).
    bool dropNow() {
        if (drop_status_) return true;
        if (canceled_) {
            drop_status_ = kStatusCanceled;
        } else if (has_deadline_ &&
                   std::chrono::steady_clock::now() >= deadline_) {
            drop_status_ = kStatusDeadline;
        }
        return drop_status_ != 0;
    }

    void readHeader() {
        auto self(shared_from_this());
        asio::async_read(
            *socket_, asio::buffer(&header_, sizeof(SessionHeader)),
            [this, self](const asio::error_code& ec, std::size_t len) {
                if (ec || len != sizeof(SessionHeader)) {
                    // Not worth a warning: the peer closed the connection,
                    // or a cancel closed it from this side.
                    if (ec.value() != asio::error::eof &&
                        ec != asio::error::operation_aborted && !canceled_) {
                        LOG(WARNING)
                            << "ServerSession::readHeader failed. Error: "
                            << ec.message() << " (value: " << ec.value() << ")"
                            << ", bytes read: " << len;
                    }
                    return;
                }
                v2_ = (header_.opcode & kOpcodeV2Flag) != 0;
                v3_ = v2_ && (header_.opcode & kOpcodeV3Flag) != 0;
#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
                if (testServerIgnoresV3()) v3_ = false;
#endif
                if (!v3_) {
                    handleRequest();
                    return;
                }
                asio::async_read(*socket_, asio::buffer(&ext_, sizeof(ext_)),
                                 [this, self](const asio::error_code& ec2,
                                              std::size_t len2) {
                                     if (ec2 || len2 != sizeof(ext_)) return;
                                     handleRequest();
                                 });
            });
    }

    void handleRequest() {
        uint8_t opcode = header_.opcode & ~kOpcodeV2Flag;
        if (v3_) {
            opcode = header_.opcode & kOpcodeMask;
            request_id_ = le64toh(ext_.request_id);
            const uint64_t remaining_us = le64toh(ext_.remaining_us);
            has_deadline_ = remaining_us != 0;
            if (has_deadline_)
                deadline_ = std::chrono::steady_clock::now() +
                            std::chrono::microseconds(remaining_us);
            if (opcode == kOpcodeCancel) {
                handleCancel();
                return;
            }
        }
        local_buffer_ = (char*)(le64toh(header_.addr));
        uint64_t size = le64toh(header_.size);
        if (validate_addr_ && !validate_addr_((uint64_t)local_buffer_, size)) {
            LOG(ERROR) << "ServerSession: remote-supplied address 0x"
                       << std::hex << (uint64_t)local_buffer_ << std::dec
                       << " with size " << size
                       << " is not within any registered buffer";
            // v2 initiators learn of the rejection; v1 initiators only see
            // the connection close (and, for small WRITEs, may have already
            // reported success — the defect v2 exists to fix).
            if (v3_ && registry_)
                registry_->finish(request_id_, kStatusAddrRejected);
            if (v2_) sendStatus(kStatusAddrRejected, nullptr);
            return;
        }
        if (v3_ && registry_) {
            request_open_ = true;
            if (!registry_->begin(request_id_, weak_from_this(),
                                  socket_->get_executor())) {
                // Canceled before its header arrived: drop all of it.
                request_open_ = false;
                canceled_ = true;
            }
        }
        if (opcode == (uint8_t)TransferRequest::WRITE) {
            readBody();
        } else if (v3_) {
            if (dropNow()) {
                finishRequest(drop_status_);
                sendStatus(drop_status_, nullptr);
                return;
            }
            sendStatus(kStatusOkV3, [this] { writeBody(); });
        } else if (v2_) {
            // READ, v2: status frame precedes the data.
            sendStatus(kStatusOk, [this] { writeBody(); });
        } else {
            writeBody();
        }
    }

    // A cancel request (its own connection): make sure nothing more of the
    // named request is applied, then answer with its final status.
    void handleCancel() {
        auto self(shared_from_this());
        if (!registry_) {
            sendCancelReply(kStatusCanceled);
            return;
        }
        ServerRequestRegistry::Running running;
        uint64_t status = 0;
        if (registry_->lookup(request_id_, &running, &status) ==
            ServerRequestRegistry::Lookup::kDone) {
            sendCancelReply(status);
            return;
        }
        auto target = running.session.lock();
        if (!target) {
            registry_->finish(request_id_, kStatusCanceled);
            sendCancelReply(kStatusCanceled);
            return;
        }
        // Cut it on its own executor (serial with its applies), then answer
        // from ours with the final status it now has (none: no answer, the
        // initiator asks again).
        auto reply_executor = socket_->get_executor();
        const uint64_t id = request_id_;
        try {
            asio::post(running.executor, [target, self, reply_executor, id] {
                target->cancelByPeer(id);
                asio::post(reply_executor, [self, id] {
                    ServerRequestRegistry::Running still_running;
                    uint64_t final_status = 0;
                    if (self->registry_->lookup(id, &still_running,
                                                &final_status) ==
                        ServerRequestRegistry::Lookup::kDone)
                        self->sendCancelReply(final_status);
                });
            });
        } catch (...) {
            // No answer: the initiator keeps the request unconfirmed.
        }
    }

    void writeBody() {
        auto self(shared_from_this());
        uint64_t size = le64toh(header_.size);
        char* addr = local_buffer_;

        size_t buffer_size =
            std::min(getChunkSize(), size - total_transferred_bytes_);
        if (buffer_size == 0) {
            // Transfer complete, wait for next request on this connection
            finishRequest(kStatusOkV3);
            start();
            return;
        }
        if (v3_ && dropNow()) {
            // Past the deadline or canceled: stop sending; the initiator
            // sees the connection close.
            finishRequest(drop_status_);
            return;
        }

        char* dram_buffer = addr + total_transferred_bytes_;
        int cuda_device = -1;

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
        cuda_device = getCudaDeviceId(addr);
        if (cuda_device >= 0) {
            dram_buffer = new char[buffer_size];
            cudaSetDevice(cuda_device);
#ifdef USE_MACA
            cudaError_t cuda_status = copyTcpCudaMemory(
                dram_buffer, addr + total_transferred_bytes_, buffer_size);
#else
            cudaError_t cuda_status =
                cudaMemcpy(dram_buffer, addr + total_transferred_bytes_,
                           buffer_size, cudaMemcpyDefault);
#endif
            if (cuda_status != cudaSuccess) {
                LOG(ERROR) << "ServerSession::writeBody failed to copy from "
                              "CUDA memory. "
                           << "Error: " << cudaGetErrorString(cuda_status);
                delete[] dram_buffer;
                return;  // Connection will be closed
            }
        }
#endif

        asio::async_write(
            *socket_, asio::buffer(dram_buffer, buffer_size),
            [this, addr, dram_buffer, cuda_device, self](
                const asio::error_code& ec, std::size_t transferred_bytes) {
#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
                if (cuda_device >= 0) {
                    delete[] dram_buffer;
                }
#endif
                if (ec) {
                    LOG(ERROR)
                        << "ServerSession::writeBody failed. "
                        << "Attempt to write data " << static_cast<void*>(addr)
                        << " using buffer " << static_cast<void*>(dram_buffer)
                        << ". Error: " << ec.message()
                        << " (value: " << ec.value() << ")";
                    return;  // Connection will be closed
                }
                total_transferred_bytes_ += transferred_bytes;
                writeBody();
            });
    }

    // Applies one received chunk to the destination; false on a CUDA error.
    bool applyChunk(char* dst, const char* src, size_t n, int cuda_device) {
#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
        if (cuda_device >= 0) {
            cudaSetDevice(cuda_device);
#ifdef USE_MACA
            cudaError_t cuda_status = copyTcpCudaMemory(dst, src, n);
#elif defined(USE_CUDA)
            // On the transport's own stream, waited for here: the chunk has
            // landed before the bounce buffer is reused and before any
            // status reports the request applied, canceled or dropped.
            cudaStream_t stream =
                device_streams_ ? device_streams_->get(cuda_device) : nullptr;
            cudaError_t cuda_status =
                stream ? cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, stream)
                       : cudaErrorInvalidResourceHandle;
            if (cuda_status == cudaSuccess)
                cuda_status = cudaStreamSynchronize(stream);
#else
            cudaError_t cuda_status =
                cudaMemcpy(dst, src, n, cudaMemcpyDefault);
#endif
            if (cuda_status != cudaSuccess) {
                LOG(ERROR) << "ServerSession::readBody failed to copy to CUDA "
                              "memory. "
                           << "Error: " << cudaGetErrorString(cuda_status);
                return false;
            }
            return true;
        }
#endif
        (void)cuda_device;
        if (dst != src) memcpy(dst, src, n);
        return true;
    }

    void readBody() {
        auto self(shared_from_this());
        uint64_t size = le64toh(header_.size);
        char* addr = local_buffer_;

        size_t buffer_size =
            std::min(getChunkSize(), size - total_transferred_bytes_);
        if (buffer_size == 0) {
            // Destination memory now holds the complete payload. Under v2,
            // acknowledge before accepting the next request — this is what
            // makes the initiator's COMPLETED mean "applied at the
            // destination" rather than "left my socket buffer". Under v3 the
            // ack is the request's final status: applied in full, or
            // dropped from the deadline / cancel on.
            if (v3_) {
                const uint64_t status =
                    drop_status_ ? drop_status_ : kStatusOkV3;
                finishRequest(status);
                sendStatus(status, [this] { start(); });
            } else if (v2_) {
                sendStatus(kStatusOk, [this] { start(); });
            } else {
                start();
            }
            return;
        }

        int cuda_device = -1;
#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
        cuda_device = getCudaDeviceId(addr);
#endif
        // Payload lands in a bounce buffer, applied after the deadline /
        // cancel check, when the device needs one or a v3 deadline is set
        // (or the request is being dropped); otherwise straight in place.
        const bool bounce = cuda_device >= 0 || (v3_ && has_deadline_) ||
                            (v3_ && (canceled_ || drop_status_));
        char* dram_buffer = addr + total_transferred_bytes_;
        if (bounce) {
            if (bounce_.size() < buffer_size) bounce_.resize(buffer_size);
            dram_buffer = bounce_.data();
        }

        asio::async_read(
            *socket_, asio::buffer(dram_buffer, buffer_size),
            [this, addr, dram_buffer, cuda_device, bounce, self](
                const asio::error_code& ec, std::size_t transferred_bytes) {
                if (ec) {
                    // If client closed connection (EOF), this is normal - don't
                    // log; nor after a cancel closed it from this side.
                    if (ec.value() != asio::error::eof &&
                        ec != asio::error::operation_aborted && !canceled_) {
                        LOG(WARNING)
                            << "ServerSession::readBody failed. "
                            << "Attempt to read data "
                            << static_cast<void*>(addr) << " using buffer "
                            << static_cast<void*>(dram_buffer)
                            << ". Error: " << ec.message()
                            << " (value: " << ec.value() << ")";
                    }
                    return;  // Connection will be closed
                }
#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
                // Tests: the chunk "arrives" this much later.
                testServerChunkDelay();
#endif
                // A chunk that arrives after the deadline, or after a
                // cancel, is dropped; its apply starts only before them.
                if (bounce && !(v3_ && dropNow())) {
                    if (!applyChunk(addr + total_transferred_bytes_,
                                    dram_buffer, transferred_bytes,
                                    cuda_device))
                        return;  // Connection will be closed
                }
                total_transferred_bytes_ += transferred_bytes;
                readBody();
            });
    }
};

// Client-side session: initiates one transfer request
struct ClientSession : public std::enable_shared_from_this<ClientSession> {
    // |unconfirmed_request|: a v3 WRITE that ended without its receiver's
    // final status (cut mid-body); the caller asks the receiver to confirm
    // (kOpcodeCancel) before the attempt counts as quiesced. 0 otherwise.
    using OnTerminal =
        std::function<void(TransferStatusEnum, bool connection_clean,
                           uint64_t unconfirmed_request)>;

    explicit ClientSession(std::shared_ptr<tcpsocket> socket, bool use_v2,
                           OnTerminal on_terminal)
        : socket_(std::move(socket)),
          v2_(use_v2),
          on_terminal_(std::move(on_terminal)) {}

    std::shared_ptr<tcpsocket> socket_;
    SessionHeader header_;
    uint64_t total_transferred_bytes_;
    char* local_buffer_;
    bool v2_;
    uint64_t status_frame_;
    // v2 WRITE runs the body stream and the ack read concurrently (one
    // async op per direction; handlers serialize on the io thread). The
    // concurrent read lets a rejection — or a v1 server's bogus payload —
    // abort a large in-flight WRITE instead of deadlocking on mutually
    // full socket buffers, and delivers rejection frames before the close.
    bool write_body_done_ = false;
    bool write_acked_ok_ = false;
    // An early negative/malformed ack can arrive while asio::async_write still
    // owns a buffer pointing into the caller's source memory. Do not publish a
    // terminal status until that body operation has completed or been
    // cancelled: callers are allowed to release the source buffer as soon as
    // the transfer becomes terminal.
    bool write_body_in_flight_ = false;
    bool write_abort_requested_ = false;
    // A v2 status frame is prompt by construction: a READ status precedes
    // any payload, and a WRITE ack follows at most one chunk's apply after
    // the body is done. The only peer that never sends one is a legacy
    // server reached through a stale v2 descriptor — and for requests
    // shorter than a frame it also keeps the connection open (it streamed
    // size < 8 bytes of "READ payload" and is waiting for our next header),
    // so without a deadline both sides wait forever. Bound that wait; the
    // default is generous so no healthy slow path can trip it, since it only
    // covers the frame itself, never payload streaming.
    static int statusFrameTimeoutSec() {
        const char* env = std::getenv("MC_TCP_STATUS_TIMEOUT_SEC");
        if (env) {
            int v = std::atoi(env);
            if (v > 0) return v;
        }
        return 30;
    }
    std::optional<asio::steady_timer> status_timer_;
    bool status_deadline_disarmed_ = false;
    // Unlike the status-frame deadline above, this is a sliding liveness
    // deadline for the whole request. It is refreshed only after an Asio
    // completion reports actual protocol or payload progress.
    static int progressTimeoutSec() {
        const char* env = std::getenv("MC_TCP_PROGRESS_TIMEOUT_SEC");
        if (env) {
            int v = std::atoi(env);
            if (v > 0) return v;
        }
        return 30;
    }
    std::optional<asio::steady_timer> progress_timer_;
    uint64_t progress_epoch_ = 0;
    bool progress_deadline_disarmed_ = true;
    bool progress_timeout_committed_ = false;
    bool terminal_reported_ = false;
    OnTerminal on_terminal_;

    // Batch limits and protocol v3 (deadline, cancel, request id).
    bool v3_ = false;
    SessionHeaderV3Ext ext_{};
    uint64_t request_id_ = 0;
    int64_t deadline_ns_ = 0;  // steady_clock ns, 0 = none
    Transport::BatchControl* control_ = nullptr;
    uint64_t aborter_id_ = 0;
    bool header_initiated_ = false;
    bool final_frame_ = false;  // v3: the receiver's final status arrived
    bool cut_ = false;
    std::optional<asio::steady_timer> total_timer_;
    std::vector<char> bounce_;

    // Before initiate(): speak v3 (the peer advertises it; implies v2), the
    // batch's deadline and its control (cancel).
    void setLimits(bool v3, int64_t deadline_ns,
                   Transport::BatchControl* control) {
        v3_ = v3 && v2_;
        deadline_ns_ = deadline_ns;
        control_ = control;
    }

    void initiate(void* buffer, uint64_t dest_addr, size_t size,
                  TransferRequest::OpCode opcode) {
        local_buffer_ = (char*)buffer;
        header_.addr = htole64(dest_addr);
        header_.size = htole64(size);
        header_.opcode = (uint8_t)opcode | (v2_ ? kOpcodeV2Flag : 0);
        total_transferred_bytes_ = 0;
        auto self(shared_from_this());
        if (limitsExpired()) {
            // Nothing is sent. Not inline: callers may hold locks here.
            asio::post(socket_->get_executor(), [this, self] {
                finalize(TransferStatusEnum::FAILED, false);
            });
            return;
        }
        if (v3_) {
            header_.opcode |= kOpcodeV3Flag;
            request_id_ = nextTcpRequestId();
            ext_.request_id = htole64(request_id_);
            uint64_t remaining_us = 0;
            if (deadline_ns_) {
                const int64_t left = deadline_ns_ - steadyNowNs();
                remaining_us = left > 1000 ? uint64_t(left / 1000) : 1;
            }
            ext_.remaining_us = htole64(remaining_us);
        }
        if (control_) {
            std::weak_ptr<ClientSession> weak = self;
            auto executor = socket_->get_executor();
            aborter_id_ = control_->addAborter([weak, executor] {
                asio::post(executor, [weak] {
                    if (auto session = weak.lock()) session->cut("canceled");
                });
            });
            if (aborter_id_ == 0) {
                asio::post(socket_->get_executor(),
                           [this, self] { cut("canceled"); });
            }
        }
        bool arm_total = deadline_ns_ != 0;
#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
        if (testClientNoCut()) arm_total = false;
#endif
        if (arm_total) {
            total_timer_.emplace(socket_->get_executor());
            total_timer_->expires_at(std::chrono::steady_clock::time_point(
                std::chrono::nanoseconds(deadline_ns_)));
            total_timer_->async_wait([this, self](const asio::error_code& ec) {
                if (ec == asio::error::operation_aborted) return;
                cut("total deadline reached");
            });
        }
        writeHeader();
    }

    bool limitsExpired() const {
        const int64_t now = steadyNowNs();
        if (control_ && control_->expired(now)) return true;
        return deadline_ns_ && now >= deadline_ns_;
    }

    // The batch's deadline passed or it was canceled: stop like a committed
    // progress timeout -- the socket is closed, and the pending operation
    // finalizes the session (a WRITE only once its body buffer is quiescent).
    void cut(const char* why) {
        if (terminal_reported_ || cut_) return;
        cut_ = true;
        progress_timeout_committed_ = true;
        progress_deadline_disarmed_ = true;
        ++progress_epoch_;
        if (progress_epoch_ == 0) ++progress_epoch_;
        if ((header_.opcode & kOpcodeMask) ==
            static_cast<uint8_t>(TransferRequest::WRITE)) {
            write_abort_requested_ = true;
        }
        VLOG(1) << "ClientSession: " << why << "; cutting the transfer";
        if (socket_ && socket_->is_open()) {
            asio::error_code ec;
            socket_->cancel(ec);
            socket_->close(ec);
        }
        if (!header_initiated_) finalize(TransferStatusEnum::FAILED, false);
    }

    void cancel() noexcept {
        cancelStatusDeadline();
        cancelProgressDeadline();
        if (!socket_) return;
        asio::error_code cancel_ec;
        socket_->cancel(cancel_ec);
        asio::error_code close_ec;
        socket_->close(close_ec);
    }

   private:
    // The status-frame deadline is separate from the sliding progress
    // deadline below. Its expiry closes the socket, but a status completion
    // already ready in the executor queue retains its original result; the
    // status handler still owns the terminal path.
    void armStatusDeadline() {
        auto self(shared_from_this());
        status_deadline_disarmed_ = false;
        status_timer_.emplace(socket_->get_executor());
        status_timer_->expires_after(
            std::chrono::seconds(statusFrameTimeoutSec()));
        status_timer_->async_wait([this, self](const asio::error_code& ec) {
            // The disarmed flag also covers an expiry that was already
            // queued when cancel() ran (cancel cannot revoke those, and by
            // then the socket may have been re-pooled).
            if (ec == asio::error::operation_aborted ||
                status_deadline_disarmed_) {
                return;
            }
            LOG(ERROR) << "ClientSession: no status frame within "
                       << statusFrameTimeoutSec()
                       << "s (peer likely speaks the legacy protocol); "
                          "dropping connection";
            if (socket_ && socket_->is_open()) {
                asio::error_code cec;
                socket_->close(cec);
            }
        });
    }

    void cancelStatusDeadline() {
        status_deadline_disarmed_ = true;
        if (status_timer_) {
            asio::error_code ec;
            status_timer_->cancel(ec);
        }
    }

    void refreshProgressDeadline() {
        if (terminal_reported_ || progress_timeout_committed_) return;
        auto self(shared_from_this());
        progress_deadline_disarmed_ = false;
        ++progress_epoch_;
        if (progress_epoch_ == 0) ++progress_epoch_;
        const uint64_t progress_epoch = progress_epoch_;
        if (!progress_timer_) progress_timer_.emplace(socket_->get_executor());
        progress_timer_->expires_after(
            std::chrono::seconds(progressTimeoutSec()));
        progress_timer_->async_wait(
            [this, self, progress_epoch](const asio::error_code& ec) {
                handleProgressDeadline(progress_epoch, ec);
            });
    }

    void handleProgressDeadline(uint64_t progress_epoch,
                                const asio::error_code& ec) {
        if (ec == asio::error::operation_aborted ||
            progress_deadline_disarmed_ || terminal_reported_ ||
            progress_epoch != progress_epoch_) {
#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
            (void)invokeSessionProgressHook(kSessionTimeoutStale, false);
#endif
            return;
        }
        progress_timeout_committed_ = true;
        progress_deadline_disarmed_ = true;
        ++progress_epoch_;
        if (progress_epoch_ == 0) ++progress_epoch_;
        if ((header_.opcode & kOpcodeMask) ==
            static_cast<uint8_t>(TransferRequest::WRITE)) {
            write_abort_requested_ = true;
        }
#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
        (void)invokeSessionProgressHook(kSessionTimeoutCommitted,
                                        write_body_in_flight_);
#endif
        LOG(ERROR) << "ClientSession: no transfer progress within "
                   << progressTimeoutSec() << "s; dropping connection";
        if (socket_ && socket_->is_open()) {
            asio::error_code cancel_ec;
            socket_->cancel(cancel_ec);
            asio::error_code close_ec;
            socket_->close(close_ec);
        }
    }

    bool acceptProgress() {
        if (progress_timeout_committed_) return false;
        refreshProgressDeadline();
        return !progress_timeout_committed_;
    }

#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
    bool acceptProgressForTest(int event) {
        const uint64_t previous_epoch = progress_epoch_;
        const int action = invokeSessionProgressHook(event, false);
        if (action == kSessionCommitTimeoutBeforeProgress)
            handleProgressDeadline(previous_epoch, asio::error_code{});
        if (!acceptProgress()) return false;
        if (action == kSessionReplayPreviousTimeoutAfterProgress)
            handleProgressDeadline(previous_epoch, asio::error_code{});
        return !progress_timeout_committed_;
    }
#endif

    void cancelProgressDeadline() {
        progress_deadline_disarmed_ = true;
        ++progress_epoch_;
        if (progress_epoch_ == 0) ++progress_epoch_;
        if (progress_timer_) {
            asio::error_code ec;
            progress_timer_->cancel(ec);
        }
    }

    // Single terminal path. The invoking Asio operation has already released
    // its buffer before entering its completion handler. The lane posts any
    // follow-up pump, so a clean socket cannot be reused inline here.
    void finalize(TransferStatusEnum status, bool clean) {
        if (terminal_reported_) return;
        terminal_reported_ = true;
        cancelStatusDeadline();
        cancelProgressDeadline();
        if (control_) control_->removeAborter(aborter_id_);
        aborter_id_ = 0;
        if (total_timer_) {
            asio::error_code ec;
            total_timer_->cancel(ec);
        }
        // A WRITE whose header went out but whose final status did not come
        // back may still be applied at the receiver: under v3 until the
        // receiver confirms the cut; under v1/v2 for as long as the receiver
        // drains what it had buffered, which it can never confirm.
        uint64_t unconfirmed = 0;
        if (status != TransferStatusEnum::COMPLETED && header_initiated_ &&
            !final_frame_ &&
            (header_.opcode & kOpcodeMask) ==
                static_cast<uint8_t>(TransferRequest::WRITE))
            unconfirmed = v3_ ? request_id_ : kUnprovableRequest;
        // v1 has no acknowledgment: a completed v1 WRITE only left this side.
        if (status == TransferStatusEnum::COMPLETED && !v2_ &&
            (header_.opcode & kOpcodeMask) ==
                static_cast<uint8_t>(TransferRequest::WRITE))
            unconfirmed = kUnprovableRequest;
#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
        (void)invokeSessionProgressHook(kSessionTerminal, clean);
#endif
        auto on_terminal = std::move(on_terminal_);
        if (on_terminal) on_terminal(status, clean, unconfirmed);
    }

    // Abort a v2 WRITE and cancel any body operation. If asio still owns the
    // current source buffer, its completion handler is responsible for
    // finalizing after the buffer is quiescent.
    void abortWrite() {
        write_abort_requested_ = true;
        if (socket_ && socket_->is_open()) {
            asio::error_code ec;
            socket_->close(ec);
        }
        if (!write_body_in_flight_) finalize(TransferStatusEnum::FAILED, false);
    }

    void writeHeader() {
        auto self(shared_from_this());
        // v3: the header and its extension in one write.
        const size_t header_bytes =
            sizeof(SessionHeader) + (v3_ ? sizeof(ext_) : 0);
        std::array<asio::const_buffer, 2> header_buffers = {
            asio::buffer(&header_, sizeof(SessionHeader)),
            asio::buffer(&ext_, v3_ ? sizeof(ext_) : 0)};
        header_initiated_ = true;
        asio::async_write(
            *socket_, header_buffers,
            [this, self, header_bytes](const asio::error_code& ec,
                                       std::size_t len) {
                if (ec || len != header_bytes) {
                    LOG(ERROR)
                        << "ClientSession::writeHeader failed. Error: "
                        << ec.message() << " (value: " << ec.value() << ")"
                        << ", bytes written: " << len;
                    finalize(TransferStatusEnum::FAILED, false);
                    return;
                }
                if (!acceptProgress()) {
                    if ((header_.opcode & kOpcodeMask) ==
                        static_cast<uint8_t>(TransferRequest::WRITE)) {
                        abortWrite();
                    } else {
                        finalize(TransferStatusEnum::FAILED, false);
                    }
                    return;
                }
                if ((header_.opcode & kOpcodeMask) ==
                    (uint8_t)TransferRequest::WRITE) {
                    if (v2_) readWriteAck();  // concurrent with the body
                    writeBody();
                } else if (v2_) {
                    readReadStatus();
                } else {
                    readBody();
                }
            });
        // Asio does not invoke the completion handler inline. Arm only after
        // initiation succeeds so a synchronous initiation exception cannot
        // leave a timer handler retaining this otherwise-abandoned session.
        refreshProgressDeadline();
    }

    // v2 READ: the server prefixes the data with a status frame.
    void readReadStatus() {
        auto self(shared_from_this());
        armStatusDeadline();
        asio::async_read(
            *socket_, asio::buffer(&status_frame_, sizeof(status_frame_)),
            [this, self](const asio::error_code& ec, std::size_t len) {
                cancelStatusDeadline();
                if (ec || len != sizeof(status_frame_)) {
                    LOG(ERROR)
                        << "ClientSession: failed to read READ status "
                           "frame. Error: "
                        << ec.message() << " (value: " << ec.value() << ")";
                    finalize(TransferStatusEnum::FAILED, false);
                    return;
                }
                if (!acceptProgress()) {
                    finalize(TransferStatusEnum::FAILED, false);
                    return;
                }
                uint64_t frame = le64toh(status_frame_);
                if (!statusFrameValid(frame)) {
                    LOG(ERROR) << "ClientSession: malformed READ status "
                                  "frame (peer likely speaks the legacy "
                                  "protocol); dropping connection";
                    finalize(TransferStatusEnum::FAILED, false);
                    return;
                }
                if (frame != (v3_ ? kStatusOkV3 : kStatusOk)) {
                    LOG(ERROR) << "ClientSession: READ rejected by server, "
                                  "status "
                               << (frame & 0xFFFFFFFFull)
                               << (v3_ && frame == kStatusOk
                                       ? " (peer did not understand protocol "
                                         "v3: stale descriptor?)"
                                       : "");
                    finalize(TransferStatusEnum::FAILED, false);
                    return;
                }
                readBody();
            });
    }

    // v2 WRITE: completion is the server's acknowledgment that the payload
    // has been applied to destination memory. Armed concurrently with the
    // body stream; a well-behaved v2 server only sends the frame after the
    // final chunk, so a frame arriving before the body is done is either a
    // rejection or a legacy peer's payload — both close the socket immediately
    // and publish failure only after the outstanding body write is quiescent.
    void readWriteAck() {
        auto self(shared_from_this());
        asio::async_read(
            *socket_, asio::buffer(&status_frame_, sizeof(status_frame_)),
            [this, self](const asio::error_code& ec, std::size_t len) {
                cancelStatusDeadline();
                if (ec || len != sizeof(status_frame_)) {
                    // The body path may have already finalized a failure and
                    // closed the socket; finalize() is idempotent (moved-from
                    // callbacks are null-checked).
                    if (ec != asio::error::operation_aborted) {
                        LOG(ERROR)
                            << "ClientSession: failed to read WRITE "
                               "ack frame. Error: "
                            << ec.message() << " (value: " << ec.value() << ")";
                    }
                    abortWrite();
                    return;
                }
                bool progress_accepted = false;
#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
                progress_accepted =
                    acceptProgressForTest(kSessionWriteAckSuccess);
#else
                progress_accepted = acceptProgress();
#endif
                if (!progress_accepted) {
                    abortWrite();
                    return;
                }
                uint64_t frame = le64toh(status_frame_);
                if (!statusFrameValid(frame)) {
                    LOG(ERROR) << "ClientSession: malformed WRITE ack frame "
                                  "(peer likely speaks the legacy protocol); "
                                  "dropping connection";
                    abortWrite();
                    return;
                }
                // A final status: nothing (more) of the request is applied.
                if ((v3_ && v3FinalStatus(frame)) ||
                    frame == kStatusAddrRejected)
                    final_frame_ = true;
                if (frame != (v3_ ? kStatusOkV3 : kStatusOk)) {
                    if (v3_ && frame == kStatusOk) {
                        LOG(ERROR) << "ClientSession: WRITE answered with a "
                                      "v2 status: the peer did not "
                                      "understand protocol v3 (stale "
                                      "descriptor?)";
                    } else if (frame == kStatusDeadline ||
                               frame == kStatusCanceled) {
                        LOG(WARNING)
                            << "ClientSession: WRITE dropped by the "
                               "receiver ("
                            << (frame == kStatusDeadline ? "deadline passed"
                                                         : "canceled")
                            << ")";
                    } else {
                        LOG(ERROR) << "ClientSession: WRITE rejected by "
                                      "server, status "
                                   << (frame & 0xFFFFFFFFull);
                    }
                    abortWrite();
                    return;
                }
                if (!write_body_done_) {
                    // The server's ack can legitimately overtake the final
                    // local write-completion handler (both become ready
                    // together for small writes; the io thread may run this
                    // handler first). Record it; the body path finalizes.
                    write_acked_ok_ = true;
                    return;
                }
                finalize(TransferStatusEnum::COMPLETED, true);
            });
    }

    void readBody() {
        auto self(shared_from_this());
        uint64_t size = le64toh(header_.size);
        char* addr = local_buffer_;

        size_t buffer_size =
            std::min(getChunkSize(), size - total_transferred_bytes_);
        if (buffer_size == 0) {
            finalize(TransferStatusEnum::COMPLETED, true);
            return;
        }

        char* dram_buffer = addr + total_transferred_bytes_;
        int cuda_device = -1;

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
        cuda_device = getCudaDeviceId(addr);
        if (cuda_device >= 0) {
            dram_buffer = new char[buffer_size];
        }
#endif
        // With a deadline, host memory is read through a bounce buffer too,
        // so that a chunk that arrives after it is never applied.
        const bool host_bounce = cuda_device < 0 && deadline_ns_ != 0;
        if (host_bounce) {
            if (bounce_.size() < buffer_size) bounce_.resize(buffer_size);
            dram_buffer = bounce_.data();
        }

        asio::async_read(
            *socket_, asio::buffer(dram_buffer, buffer_size),
            [this, addr, dram_buffer, cuda_device, host_bounce, self](
                const asio::error_code& ec, std::size_t transferred_bytes) {
                if (ec) {
                    LOG_IF(ERROR, !cut_)
                        << "ClientSession::readBody failed. "
                        << "Attempt to read data " << static_cast<void*>(addr)
                        << " using buffer " << static_cast<void*>(dram_buffer)
                        << ". Error: " << ec.message()
                        << " (value: " << ec.value() << ")";
#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
                    if (cuda_device >= 0) delete[] dram_buffer;
#endif
                    finalize(TransferStatusEnum::FAILED, false);
                    return;
                }

                bool progress_accepted = !progress_timeout_committed_;
                if (progress_accepted && transferred_bytes > 0) {
#ifdef MOONCAKE_TCP_TRANSPORT_TEST_HOOKS
                    progress_accepted =
                        acceptProgressForTest(kSessionReadBodySuccess);
#else
                    progress_accepted = acceptProgress();
#endif
                }
                if (!progress_accepted || (deadline_ns_ && limitsExpired())) {
                // Past the deadline: this chunk is not applied.
#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
                    if (cuda_device >= 0) delete[] dram_buffer;
#endif
                    finalize(TransferStatusEnum::FAILED, false);
                    return;
                }
                if (host_bounce)
                    memcpy(addr + total_transferred_bytes_, dram_buffer,
                           transferred_bytes);

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
                if (cuda_device >= 0) {
                    cudaSetDevice(cuda_device);
#ifdef USE_MACA
                    cudaError_t cuda_status =
                        copyTcpCudaMemory(addr + total_transferred_bytes_,
                                          dram_buffer, transferred_bytes);
#else
                    cudaError_t cuda_status =
                        cudaMemcpy(addr + total_transferred_bytes_, dram_buffer,
                                   transferred_bytes, cudaMemcpyDefault);
#endif
                    if (cuda_status != cudaSuccess) {
                        LOG(ERROR)
                            << "ClientSession::readBody failed to copy to CUDA "
                               "memory. "
                            << "Error: " << cudaGetErrorString(cuda_status);
                        delete[] dram_buffer;
                        finalize(TransferStatusEnum::FAILED, false);
                        return;
                    }
                    delete[] dram_buffer;
                }
#endif
                total_transferred_bytes_ += transferred_bytes;
                readBody();
            });
    }

    void writeBody() {
        auto self(shared_from_this());
        uint64_t size = le64toh(header_.size);
        char* addr = local_buffer_;

        size_t buffer_size =
            std::min(getChunkSize(), size - total_transferred_bytes_);
        if (buffer_size == 0) {
            if (v2_) {
                if (write_abort_requested_) {
                    finalize(TransferStatusEnum::FAILED, false);
                    return;
                }
                // Completion comes from the server's acknowledgment, whose
                // read is already in flight (armed in writeHeader) and may
                // have finished first.
                write_body_done_ = true;
                if (write_acked_ok_) {
                    finalize(TransferStatusEnum::COMPLETED, true);
                } else {
                    // From here a well-behaved server owes at most one
                    // chunk's apply plus the frame; a legacy peer behind a
                    // stale descriptor may owe nothing, ever.
                    armStatusDeadline();
                }
            } else {
                // v1: no acknowledgment exists in the protocol; this only
                // means the payload left the initiator (#2086). (Nor can a
                // v1 receiver confirm anything: such a COMPLETED counts as
                // quiesced, as before.)
                finalize(TransferStatusEnum::COMPLETED, true);
            }
            return;
        }

        char* dram_buffer = addr + total_transferred_bytes_;
        int cuda_device = -1;

#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) ||  \
    defined(USE_MLU) || defined(USE_MACA) || defined(USE_HYGON) || \
    defined(USE_COREX)
        cuda_device = getCudaDeviceId(addr);
        if (cuda_device >= 0) {
            dram_buffer = new char[buffer_size];
            cudaSetDevice(cuda_device);
#ifdef USE_MACA
            cudaError_t cuda_status = copyTcpCudaMemory(
                dram_buffer, addr + total_transferred_bytes_, buffer_size);
#else
            cudaError_t cuda_status =
                cudaMemcpy(dram_buffer, addr + total_transferred_bytes_,
                           buffer_size, cudaMemcpyDefault);
#endif
            if (cuda_status != cudaSuccess) {
                LOG(ERROR) << "ClientSession::writeBody failed to copy from "
                              "CUDA memory. "
                           << "Error: " << cudaGetErrorString(cuda_status);
                delete[] dram_buffer;
                abortWrite();
                return;
            }
        }
#endif

        write_body_in_flight_ = true;
        asio::async_write(
            *socket_, asio::buffer(dram_buffer, buffer_size),
            [this, addr, dram_buffer, cuda_device, self](
                const asio::error_code& ec, std::size_t transferred_bytes) {
                write_body_in_flight_ = false;
                if (cuda_device >= 0) {
                    delete[] dram_buffer;
                }
                if (ec && !cut_) {
                    LOG(ERROR)
                        << "ClientSession::writeBody failed. "
                        << "Attempt to write data " << static_cast<void*>(addr)
                        << " using buffer " << static_cast<void*>(dram_buffer)
                        << ". Error: " << ec.message()
                        << " (value: " << ec.value() << ")";
                    abortWrite();
                    return;
                }
                if (ec) {  // cut (deadline or cancel): expected
                    abortWrite();
                    return;
                }
                if (write_abort_requested_) {
                    // An early ACK failure or a committed progress timeout
                    // closed the socket while this operation still owned the
                    // caller's source buffer. It is safe to publish failure
                    // now that the handler has run.
                    finalize(TransferStatusEnum::FAILED, false);
                    return;
                }
                if (transferred_bytes > 0 && !acceptProgress()) {
                    finalize(TransferStatusEnum::FAILED, false);
                    return;
                }
                total_transferred_bytes_ += transferred_bytes;
                writeBody();
            });
    }
};

struct TcpContext {
    TcpContext(short port, size_t io_threads, ValidateAddrFn validate_addr)
        : io_contexts(makeIoContexts(io_threads)),
          io_context(*io_contexts.front()),
          acceptor(io_context),
          validate_addr_(std::move(validate_addr)) {
        std::error_code ec;
        asio::ip::tcp::endpoint endpoint(asio::ip::tcp::v6(), port);

        acceptor.open(endpoint.protocol(), ec);
        if (!ec) {
            acceptor.set_option(asio::ip::v6_only(false), ec);
            if (!ec) {
                acceptor.set_option(
                    asio::ip::tcp::acceptor::reuse_address(true));
                acceptor.bind(endpoint, ec);
                if (!ec) {
                    acceptor.listen();
                    return;
                }
            }
            acceptor.close();
        }
        LOG(ERROR) << "Failed to set up IPv6 dual-stack listener: "
                   << ec.message() << " (error code: " << ec.value() << ")";
        asio::ip::tcp::endpoint endpoint_v4(asio::ip::tcp::v4(), port);
        acceptor.open(endpoint_v4.protocol());
        acceptor.set_option(asio::ip::tcp::acceptor::reuse_address(true));
        acceptor.bind(endpoint_v4);
        acceptor.listen();
    }

    static std::vector<std::unique_ptr<asio::io_context>> makeIoContexts(
        size_t count) {
        std::vector<std::unique_ptr<asio::io_context>> contexts;
        contexts.reserve(std::max<size_t>(count, 1));
        for (size_t i = 0; i < std::max<size_t>(count, 1); ++i) {
            // One thread runs each context: concurrency hint 1 lets Asio
            // skip internal locking it would otherwise need.
            contexts.push_back(std::make_unique<asio::io_context>(1));
        }
        return contexts;
    }

    // Accepted connections are spread round-robin over the io contexts so
    // that server sessions for different connections run on different
    // threads. Each session is started on its own socket's executor.
    asio::io_context& nextIoContext() {
        const size_t index =
            next_io_context.fetch_add(1, std::memory_order_relaxed);
        return *io_contexts[index % io_contexts.size()];
    }

    void doAccept() {
        // The accepted socket is bound to the chosen context's executor.
        const asio::any_io_executor socket_executor =
            nextIoContext().get_executor();
        acceptor.async_accept(socket_executor, [this](asio::error_code ec,
                                                      tcpsocket socket) {
            if (!ec) {
                asio::error_code nodelay_ec;
                socket.set_option(asio::ip::tcp::no_delay(true), nodelay_ec);
                auto socket_ptr =
                    std::make_shared<tcpsocket>(std::move(socket));
                auto session = std::make_shared<ServerSession>(
                    socket_ptr, validate_addr_, registry, device_streams);
                try {
                    asio::post(socket_ptr->get_executor(),
                               [session] { session->start(); });
                } catch (const std::exception& e) {
                    LOG(ERROR) << "TcpContext: failed to start a server "
                                  "session: "
                               << e.what();
                }
            }
            doAccept();
        });
    }

    std::vector<std::unique_ptr<asio::io_context>> io_contexts;
    // The first context; it also hosts the acceptor.
    asio::io_context& io_context;
    asio::ip::tcp::acceptor acceptor;
    ValidateAddrFn validate_addr_;
    std::atomic<size_t> next_io_context{0};
    // v3 requests seen by this server, for cancellation by request id.
    std::shared_ptr<ServerRequestRegistry> registry =
        std::make_shared<ServerRequestRegistry>();
    std::shared_ptr<TcpDeviceStreams> device_streams =
        std::make_shared<TcpDeviceStreams>();
};
