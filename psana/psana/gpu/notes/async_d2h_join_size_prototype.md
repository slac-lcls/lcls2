# Async D2H Join-Size Prototype

## Goal

Prototype a `join_size`-based D2H path so we can measure the performance of
copying GPU results to CPU in larger asynchronous chunks, without changing the
current `GpuEvents` implementation first.

This is intentionally a prototype.  The purpose is to answer the performance
question before committing to a deeper integration:

```text
Can we keep the fast GPU read/calib path, retain N GPU results, then copy those
N results to CPU as one asynchronous join operation?
```

## Evidence For This Design

See [d2h_interval_bandwidth_results.md](d2h_interval_bandwidth_results.md).

The key measurements from that note are:

| Run | Loop time | Rate | D2H calls | D2H time |
| --- | ---: | ---: | ---: | ---: |
| CPU path | 419.61 s | 38.1 Hz | 0 | 0 |
| GPU, no D2H | 159.47 s | 100.3 Hz | 0 | 0 |
| GPU, D2H every 100 events | 162.10 s | 98.7 Hz | 160 | 2.63 s |
| GPU, D2H every 10 events | 185.39 s | 86.3 Hz | 1600 | 25.90 s |
| GPU, D2H every event | 404.16 s | 39.6 Hz | 16000 | 246.37 s |

Subtracting measured D2H time from the GPU runs gives a nearly constant base
GPU time:

```text
jn100: 162.10 - 2.63   = 159.47 s
jn10:  185.39 - 25.90  = 159.49 s
jn1:   404.16 - 246.37 = 157.79 s
```

This shows that current `.on_cpu` D2H is synchronous and additive.  It is not
overlapped with later read/H2D/compute work.

The same note also shows that the no-D2H GPU path drives higher NIC read
bandwidth than the CPU path, even without GDS:

| Run | Active NIC recv avg | NIC recv max |
| --- | ---: | ---: |
| CPU | 1.27 GB/s | 1.51 GB/s |
| GPU, no D2H | 2.31 GB/s | 3.60 GB/s |
| GPU, D2H every event | 1.30 GB/s | 1.58 GB/s |

So the immediate target is clear:

```text
keep the high-throughput GPU read/calib path
avoid a blocking per-event D2H join
copy larger batches of GPU results asynchronously
```

## Terminology

`gpu_d2h_interval` is a sampling knob:

```text
gpu_d2h_interval=100 means copy 1 event every 100 events
```

`gpu_join_size` should be a production join knob:

```text
gpu_join_size=100 means copy all 100 events, but as one joined batch
```

These two knobs are not equivalent and should remain separate.

## Current Path

Current user-facing path:

```python
for ctx in run.events():
    calib_gpu = ctx.get("calib").on_gpu    # CuPy, no D2H
    calib_cpu = ctx.get("calib").on_cpu    # synchronous D2H
```

`ctx.get("calib").on_cpu` currently:

```text
1. synchronizes the producing CUDA stream
2. calls CuPy .get()
3. blocks Python until the D2H copy is complete
```

That means the BD rank is not asking EB for the next batch while the D2H copy
is happening.

## Prototype Design

Add a user-space async D2H joiner, probably in:

```text
psana/psana/gpu/gpu_d2h_joiner.py
```

The prototype should work outside `GpuEvents`:

```python
from psana.gpu.gpu_d2h_joiner import AsyncD2HJoiner

joiner = AsyncD2HJoiner(join_size=100, max_inflight=2)

for ctx in run.events():
    pending = joiner.add(ctx.timestamp, ctx.get("calib").on_gpu)
    if pending is not None:
        ready = pending.wait()
        timestamps = ready.timestamps
        calib_cpu = ready.array
        process(timestamps, calib_cpu)

final = joiner.flush()
if final is not None:
    ready = final.wait()
    process(ready.timestamps, ready.array)
```

The core behavior:

```text
1. Allocate a GPU join buffer:       (join_size, *result_shape)
2. Allocate pinned host buffer:      (join_size, *result_shape)
3. For each event:
      D2D copy event result into join_buffer_gpu[index]
4. When join_size events are accumulated:
      launch async D2H join_buffer_gpu[:n] -> pinned_host[:n]
      record a CUDA event
      immediately switch to another join slot
5. CPU calls wait() only when it wants to consume the joined block
```

## Critical Safety Rule

Do not store `ctx.get("calib").on_gpu` references for later use.

Current GPU results are views into `EventPool` slot buffers.  Those buffers are
reused.  The joiner must immediately copy each event result into its own join
buffer:

```python
join_buffer_gpu[index][...] = calib_gpu
```

This is a device-to-device copy.  It keeps the event result safe after the
`EventPool` slot is recycled.

## Async D2H Mechanics

Use pinned host memory and a non-blocking CUDA stream:

```python
import cupy as cp
import cupyx

copy_stream = cp.cuda.Stream(non_blocking=True)
done = cp.cuda.Event()

host = cupyx.empty_pinned(join_shape, dtype=calib_gpu.dtype)

with copy_stream:
    gpu_join_slot[:n].get(out=host[:n], stream=copy_stream, blocking=False)
    done.record(copy_stream)
```

The CPU must not touch `host[:n]` until:

```python
done.synchronize()
```

This gives us the first measurement of whether D2H can overlap with later
read/H2D/compute in the application loop.

## Memory Estimate

For a full Jungfrau calibrated result:

```text
32 * 512 * 1024 * float32 = 64 MiB/event
```

Join buffer size per BD rank:

| join_size | GPU buffer size | pinned host buffer size |
| ---: | ---: | ---: |
| 16 | 1.0 GiB | 1.0 GiB |
| 32 | 2.0 GiB | 2.0 GiB |
| 64 | 4.0 GiB | 4.0 GiB |
| 100 | 6.4 GiB | 6.4 GiB |
| 256 | 16.0 GiB | 16.0 GiB |

With `max_inflight=2`, double the join-buffer memory.

For one BD per 40 GB A100, `join_size=100` is a reasonable prototype target.
For multiple BDs sharing one GPU, start lower:

```text
join_size=16 or join_size=32 per BD
```

## Suggested Prototype API

```python
class AsyncD2HJoiner:
    def __init__(self, join_size, max_inflight=2, dtype=None):
        ...

    def add(self, timestamp, array_gpu):
        """Add one GPU result.

        Returns a PendingD2HJoin when a slot has been submitted for async D2H,
        otherwise returns None.
        """

    def flush(self):
        """Submit any partial slot and return PendingD2HJoin or None."""


class PendingD2HJoin:
    def ready(self):
        """Return True if D2H has completed."""

    def wait(self):
        """Synchronize the D2H event and return JoinedD2HResult."""


class JoinedD2HResult:
    timestamps: np.ndarray
    array: np.ndarray
    n_events: int
```

The first implementation can support only a single CuPy array per event.  That
is enough for the performance test:

```python
ctx.get("calib").on_gpu
```

Later versions can support multiple arrays, CPU metadata, or segmented
Jungfrau routing.

## ds_count_events.py Test Hook

Add a separate option:

```text
--gpu_join_size N
--gpu_join_inflight M
```

Initial behavior:

```text
--gpu_join_size 0      disabled
--gpu_join_size N      copy every event in async N-event chunks
```

Keep this mutually exclusive with `--gpu_d2h_interval` for the first test:

```text
gpu_d2h_interval: validation sampling, synchronous, copies 1 event every N
gpu_join_size:    performance join, async, copies every event in N-sized chunks
```

Example command:

```bash
mpirun -n 3 python psana/psana/debugtools/ds_count_events.py \
  -e mfx101210926 -r 387 \
  --dir /sdf/data/lcls/drpsrcf/ffb/mfx/mfx101210926/xtc \
  --gpu_det jungfrau \
  --batch_size 1 \
  --gpu_pool_depth 2 \
  --max_events 16000 \
  --gpu_join_size 100 \
  --gpu_join_inflight 2
```

## Measurements To Collect

Compare:

```text
CPU baseline
GPU no D2H
GPU gpu_d2h_interval=1
GPU gpu_join_size=16
GPU gpu_join_size=32
GPU gpu_join_size=100
```

Record:

```text
Loop time
Total event rate
join D2H bytes
join D2H submit time
join D2H wait time
effective D2H bandwidth
NIC recv average/max from net_bandwidth.py
GPU memory used
pinned host memory used
```

Expected result:

```text
gpu_join_size=N should be faster than gpu_d2h_interval=1
if the D2H transfer overlaps with later read/H2D/compute.
```

If it is not faster, the likely explanations are:

```text
D2H bandwidth saturates PCIe and blocks H2D/read fallback traffic
copy stream is implicitly synchronized by the producer stream
CPU touches the pinned host buffer too early
join buffers are too large and cause memory pressure
```

## Relationship To EventJoiner Design

This note is narrower than
[event_joiner_implementation.md](event_joiner_implementation.md).

`event_joiner_implementation.md` describes a general CPU/GPU semantic joiner.
This prototype is only for the performance question:

```text
Can async batched D2H recover most of the no-D2H GPU throughput while still
moving all requested GPU results to CPU?
```

After the prototype answers that, the useful pieces can be folded into the
general `EventJoiner` implementation.

## Why Not Change GpuEvents First?

Changing `GpuEvents` first would mix several questions:

```text
event iterator semantics
result lifetime
join API
CPU/GPU partial detector routing
async D2H mechanics
memory policy for multi-BD GPU sharing
```

The prototype keeps those separate.  It uses the current public behavior:

```python
ctx.get("calib").on_gpu
```

and adds only an external consumer that copies those results into a longer-lived
join buffer.

If the prototype shows a meaningful speedup, then the next step is to integrate
the same buffering and async copy policy into the real GPU result/join API.
