# NVLink Proxy Transport

The `nvlink_proxy` transport moves data GPU to GPU between Transfer Engine
instances that run on the same node but in containers that each see only
their own GPU, which is the usual shape when a Kubernetes device plugin
allocates GPUs per pod.

## Why a proxy

Same-node GPU transports (`nvlink_intra`, legacy CUDA IPC) need the importing
process to see the exporting GPU. With per-container GPU isolation,
`cudaIpcOpenMemHandle` fails (`invalid argument`), POSIX-fd cuMem handles fail
with `CUDA_ERROR_INVALID_DEVICE`, and fabric handles need an IMEX daemon. The
only remaining same-node path would be TCP/RDMA loopback through host memory.

`nvlink_proxy` adds a small node-local daemon, `mooncake_nvlink_proxy`, that
runs in a container which sees every GPU of the node. Engines register their
`cudaMalloc` blocks with it (the 64-byte legacy IPC handle is sent as plain
data over a unix socket) and ask it to copy between registered blocks. The
daemon enables peer access between all GPU pairs at start, opens both handles
in the context of the GPU that receives the data and copies with
`cudaMemcpyBatchAsync`, so the data moves over NVLink or PCIe P2P.

The transport is additive: it is installed next to the base transport (RDMA
or TCP), and every request it cannot serve uses the base transport.
Cross-node transfers are unaffected.

## Build

```bash
cmake .. -DUSE_CUDA=ON -DENABLE_MULTI_PROTOCOL=ON -DUSE_NVLINK_PROXY=ON [other flags]
```

The build produces `mooncake_nvlink_proxy` (CUDA runtime linked statically;
only the driver is needed at run time). `scripts/build_wheel.sh` ships it in
the wheel with the console entry point `mooncake_nvlink_proxy`. The workflow
`.github/workflows/release-nvlink-proxy.yaml` builds such a wheel.

## Deployment

1. One daemon per node, in a container with no GPU request and
   `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility`:

   ```bash
   mooncake_nvlink_proxy --socket /run/mooncake-nvlink-proxy/proxy.sock
   ```

   The GPUs must be in compute mode `DEFAULT`: in `EXCLUSIVE_PROCESS` mode the
   daemon cannot create a context on a GPU an engine already uses, so it
   refuses the registration and the engines keep using the base transport
   (the daemon logs the compute mode of every GPU at start). The daemon
   creates a context on a GPU only when the first block on it is registered
   (`--eager-init` initializes all GPUs at start).

   The socket directory is shared with the engine containers, e.g. through a
   `hostPath` volume. Access to the socket allows copying between registered
   buffers, so treat the directory as a trust boundary. `--ping` exits 0 when
   the daemon answers and can serve as a liveness probe.

2. Engines (all peers must run a build with `USE_NVLINK_PROXY=ON`):

   | Variable | Meaning |
   |---|---|
   | `MC_NVLINK_PROXY_SOCKET` | Daemon socket. When set, `nvlink_proxy` is installed in addition to the base transport, also with `MC_FORCE_TCP=1`. |
   | `MC_NODE_ID` | Locality key, published in the segment metadata. Default: `NODE_NAME`, else the hostname. In Kubernetes set `NODE_NAME` from `spec.nodeName`; pod hostnames differ, so without it the proxy is never used. |
   | `MC_NVLINK_PROXY_TIMEOUT_MS` | Upper bound for one daemon round trip (default 30000). The daemon refuses to start a copy after half of it. |
   | `MC_NVLINK_PROXY_STATS_INTERVAL` | Seconds between counter log lines (default 60, 0 = off). |

   The Python API is unchanged.

## Routing and fallback

- Device memory is published under both `nvlink_proxy` and the base transport;
  host memory only under the base transport (`register_memory` returns 0).
  Device memory that cannot be exported with `cudaIpcGetMemHandle` (for
  example virtual-memory-management allocations) also stays on the base
  transport.
- A request uses `nvlink_proxy` when the target buffer was published with the
  same `MC_NODE_ID`, the local buffer is registered with the daemon and the
  daemon is healthy. Otherwise it uses the base transport.
- Any proxy failure (daemon down, unknown registration, CUDA error, timeout)
  resubmits the affected requests to the base transport. Failures are logged
  at most every 5 seconds and counted.
- Registrations are keyed by a random per-engine client id and the block base,
  both published in the segment metadata. When the daemon restarts, engines
  reconnect within about a second and re-register the same keys, so peers'
  cached metadata stays valid.
- The daemon drops an engine's registrations when its control connection
  closes; copies in flight keep the mappings alive until they finish. On an
  unrecoverable CUDA error the daemon exits so that it can be restarted.

## Counters

Engine side, logged by each engine every `MC_NVLINK_PROXY_STATS_INTERVAL`
seconds when they changed (and at shutdown):

```
nvlink_proxy stats: healthy=1 proxied_requests=... proxied_bytes=... proxied_batches=...
  avg_batch_us=... max_batch_us=... fallback_requests=... (daemon_unavailable=...
  daemon_error=... not_servable=...) remote_node_requests=... registered_blocks=N/M connects=...
```

- `proxied_*`: requests, bytes and daemon round trips served by the proxy;
  `avg/max_batch_us` is the round-trip latency of one batch.
- `fallback_requests`: same-node GPU requests that used the base transport
  instead, split by reason.
- `remote_node_requests`: requests to buffers on another node (normal base
  transport traffic).

Daemon side, logged every `--stats-interval` seconds when they changed, or on
demand with `mooncake_nvlink_proxy --socket <path> --stats`: active clients,
registrations, copy requests / entries / bytes / failures, average and maximum
copy latency, IPC handle opens.

## Wire protocol

Unix stream socket, little-endian packed structs, see
`mooncake-transfer-engine/include/transport/nvlink_proxy_transport/nvlink_proxy_protocol.h`.
Messages: `HELLO` (client id, control flag; reply carries the daemon epoch),
`REGISTER` (block base, size, GPU UUID, IPC handle), `UNREGISTER`, `COPY`
(list of `(src client, src base, src offset, dst client, dst base,
dst offset, length)` plus a start budget; replied after the copies completed),
`PING`, `STATS`.

## Testing

`mooncake-transfer-engine/tests/nvlink_proxy_e2e_test.py` runs a target and an
initiator process (e.g. two one-GPU containers on one node plus the daemon) and
verifies large and paged writes, reads, sub-allocated buffers, guard regions
and host memory byte by byte. `nvlink_proxy_transport_test` covers the
protocol helpers and the no-daemon path without GPUs.
