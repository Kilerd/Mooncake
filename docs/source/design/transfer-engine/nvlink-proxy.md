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
   | `MC_NVLINK_PROXY_TIMEOUT_MS` | Request budget (default 30000): upper bound for one daemon round trip and for completing a request on the base transport after a fallback. The daemon refuses to start a copy after half of it. |
   | `MC_NVLINK_PROXY_RECONNECT_WAIT_MS` | How long same-node requests wait for an unavailable daemon before they use the base transport (default: half of the request budget). |
   | `MC_NVLINK_PROXY_FALLBACK_INFLIGHT` | Maximum requests the proxy has in flight on the base transport at a time, across all threads (default 512; with TCP at most half of `MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER`). |
   | `MC_NVLINK_PROXY_STATS_INTERVAL` | Seconds between counter log lines (default 60, 0 = off). |

   The Python API is unchanged.

## Batches of many small copies

KV caches with small pages produce batches of thousands of entries of a few
KB each, whose source and destination runs are short. Submitting every entry
to the copy engines costs microseconds per entry, so the daemon:

1. merges consecutive entries that are contiguous in both source and
   destination;
2. copies entries shorter than `--gather-threshold` bytes (default 128 KiB)
   with one gather/scatter kernel launch per GPU and batch. The entry table is
   uploaded to the GPU, each thread block copies whole entries with 16-byte
   accesses where the relative alignment of source and destination allows
   (8/4/2/1-byte otherwise). Larger entries keep using `cudaMemcpyBatchAsync`.

`--gather-on dst` (default) runs the kernel on the GPU that receives the data,
reading through the peer mapping; `--gather-on src` runs it on the sending GPU,
writing through the mapping. `--gather-threshold 0` and `--no-coalesce` restore
the copy-engine-only behaviour. If a GPU has no usable kernel image the daemon
logs it and uses the copy engines for that GPU.

On the client, routing a request no longer scans every buffer of the target
segment: the buffers covering the last target address are cached per thread
together with the exact address range in which that set is unchanged.

## Routing and fallback

- Device memory is published under both `nvlink_proxy` and the base transport;
  host memory only under the base transport (`register_memory` returns 0).
  Device memory that cannot be exported with `cudaIpcGetMemHandle` (for
  example virtual-memory-management allocations) also stays on the base
  transport.
- A request uses `nvlink_proxy` when the target buffer was published with the
  same `MC_NODE_ID`, the local buffer is registered with the daemon and the
  daemon is healthy. Otherwise it uses the base transport.
- While the daemon is unavailable (restart, crash), same-node GPU requests
  stay with the proxy: in-flight and new batches wait for the daemon to come
  back and are then re-sent through it. Engines retry the connection every
  100-400 ms and re-register the same keys. A batch waits at most
  `MC_NVLINK_PROXY_RECONNECT_WAIT_MS`, and never beyond that window after the
  outage began, so a daemon that stays away does not delay every request.
- Right after a daemon (re)start the peer may not have re-registered yet: a
  COPY that names a block unknown to a daemon this engine connected to within
  the reconnect window is retried with backoff instead of falling back.
- Requests that still need the base transport (daemon away for longer than the
  window, or a non-retryable error such as a CUDA error) are moved there with
  backpressure: at most `MC_NVLINK_PROXY_FALLBACK_INFLIGHT` requests in flight,
  rejected requests (e.g. a full TCP lane queue) retried with backoff until
  the request budget is spent. Fallbacks are logged at most every 5 seconds
  and counted.
- Registrations are keyed by a random per-engine client id and the block base,
  both published in the segment metadata, so peers' cached metadata stays
  valid across daemon restarts.
- The daemon drops an engine's registrations when its control connection
  closes; copies in flight keep the mappings alive until they finish. On
  SIGTERM it stops accepting, lets running copies finish (up to
  `--drain-timeout-ms`, default 10 s) and exits; on an unrecoverable CUDA error
  it exits so that it can be restarted.

## P2P integrity self-test

Some platforms report peer access between two GPUs (`cudaDeviceCanAccessPeer`,
`nvidia-smi topo -p2p`) and complete peer copies without any error, yet the
data never arrives or arrives corrupted. The daemon therefore verifies a GPU
pair before it serves copies between them.

- The test never runs inside the daemon: the daemon re-executes its own binary
  in a child process (`posix_spawn`, nothing CUDA is forked) for the pairs to
  test and reads one verdict per direction from a pipe. A child that makes no
  progress for `--p2p-selftest-timeout` seconds (default 30) is killed. A pair
  the child was testing when it died (for instance on an unrecoverable CUDA
  error) or hung is failed; pairs it had not reached are tested by a new
  child. The daemon keeps serving throughout.
- For both directions of a pair the child copies patterns with the production
  code paths -- `cudaMemcpyBatchAsync` on the receiving GPU's stream (an entry
  of at least the gather threshold, whole 2 KiB rows, small and odd lengths,
  misaligned offsets, runs of up to 2 MiB) and the gather kernel on the GPU
  `--gather-on` selects (single 2 KiB rows, small and byte-misaligned entries)
  -- three iterations each. The destination buffer is filled with a different
  poison before every pass and read back in full: copied ranges must hold the
  source GPU's bytes and everything else the poison, so dropped, corrupted and
  stray writes are all caught. Each tested GPU gets two buffers of up to
  4 MiB (more if the gather threshold needs it; smaller sizes are tried when
  memory is short).
- Only a data mismatch (or the child dying or hanging on the pair) fails a
  pair. A test that could not run (no memory, a CUDA error, a failed pattern
  upload) leaves the pair untested; it is tested again at the next
  registration on either GPU. A direction without peer access is not judged:
  its copies are refused as no-P2P anyway.
- When it runs: when an engine registers its first block on a GPU, the
  GPU's pairs with the other GPUs that engines use are tested in the
  background unless they already passed or failed. By default every pair of
  GPUs in compute mode Default is also tested right after the socket is up
  (a node audit); `--no-startup-selftest` skips that, so GPUs without engines
  are never touched -- recommended in production.
- `--p2p-selftest=enforce` (default): a pair is refused from the moment it is
  requested until its test passes, for good if it failed; while it is
  untested it stays refused. `warn` only logs; `off` skips the test.
- `--deny-peer <bus id>` (repeatable) denies every pair with that GPU and
  `--deny-pair <bus id>,<bus id>` one pair, without testing. Bus ids not
  present on the node are ignored with a warning.

A copy between the GPUs of a denied pair is refused with `kNoPeerAccess`
while its entries are resolved, before anything is merged or mapped; the
engine moves the request to its base transport (counted under
`fallback_requests` / `daemon_error`). A copy within one GPU is never denied.

The child logs one line per direction and the daemon one verdict per pair,
e.g.

```
P2P self-test gpu 0 (0000:1a:00) -> gpu 1 (0000:3d:00): copy engines OK (3 x 4096 KiB, 21000 MB/s), gather kernel on gpu 1 OK: PASS
P2P self-test gpu 0 (0000:1a:00) <-> gpu 1 (0000:3d:00): passed -> allowed
P2P self-test gpu 2 (0000:5e:00) -> gpu 3 (0000:b1:00): copy engines CORRUPT (iteration 0: ... bytes wrong ...), ...: FAIL
P2P self-test gpu 2 (0000:5e:00) <-> gpu 3 (0000:b1:00): FAILED -> DENIED, copies between them use the engines' base transport
```

followed by a summary per batch and, when every pair tested so far failed,
`P2P unusable on this node: N/N GPU pairs tested so far failed the self-test`.

## Counters

Engine side, logged by each engine every `MC_NVLINK_PROXY_STATS_INTERVAL`
seconds when they changed (and at shutdown):

```
nvlink_proxy stats: healthy=1 proxied_requests=... proxied_bytes=... proxied_batches=...
  avg_batch_us=... max_batch_us=... held_requests=... max_hold_ms=... retried_requests=...
  fallback_requests=... (daemon_unavailable=... daemon_error=... not_servable=...)
  fallback_retries=... failed_requests=... remote_node_requests=...
  registered_blocks=N/M connects=...
```

- `proxied_*`: requests, bytes and daemon round trips served by the proxy;
  `avg/max_batch_us` is the round-trip latency of one batch.
- `held_requests` / `max_hold_ms`: requests that waited for the daemon and the
  longest wait of a batch; `retried_requests`: requests re-sent to the daemon
  (after a restart or while a peer re-registered).
- `fallback_requests`: same-node GPU requests that used the base transport
  instead, split by reason; `fallback_retries`: base-transport attempts that
  were retried (e.g. full TCP queue); `failed_requests`: requests that failed
  within the request budget.
- `remote_node_requests`: requests to buffers on another node (normal base
  transport traffic).

Daemon side, logged every `--stats-interval` seconds when they changed, or on
demand with `mooncake_nvlink_proxy --socket <path> --stats`: active clients,
registrations, copy requests / entries / bytes / failures, average and maximum
copy latency split into `avg_plan_us` (validation, merging, mapping) and
`avg_exec_us` (GPU work), `coalesced_entries`, `kernel_entries/bytes` and
`ce_entries/bytes` (copy engines), IPC handle opens, the self-test mode,
`denied_pairs` (the number of denied ordered `src>dst` directions -- a denied
GPU pair counts twice -- followed by their bus ids) and
`denied_copy_requests` (copies refused because their pair is denied; not
counted as failures).

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
protocol helpers and the no-daemon path without GPUs;
`nvlink_proxy_policy_test` the self-test plans, the verifier, bus id parsing
and the deny matrix.
