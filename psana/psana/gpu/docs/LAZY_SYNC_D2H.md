# Lazy-Sync D2H and GPU Slot Lifetime

## Status and scope

This document describes the implemented psana2 GPU result-lifetime model.
The current implementation supports Jungfrau calibration and two result
consumption modes:

1. Automatic D2H, enabled with `gpu_d2h_chunk_size > 0`.
2. External GPU consumption, selected with `gpu_d2h_chunk_size = 0`.

These modes have different retirement ordering because they expose different
result lifetimes. Automatic-D2H results can be detached from the execution
slot before delivery. External GPU results remain slot-backed while the user
registers downstream work.

The central invariant is:

> An execution slot cannot be overwritten until every terminal consumer of
> its current contents has completed.

Advancing a Python generator is not a CUDA completion signal.

---

## Configuration

```python
ds = DataSource(
    ...,
    gpu_det="jungfrau",
    n_gpu_streams=2,
    gpu_d2h_chunk_size=2,
    gpu_memory_budget_gb=15,
)
```

The controls have separate meanings:

| Control | Meaning |
|---|---|
| `n_gpu_streams` | Number of reusable GPU execution slots (`EventPool` depth) |
| `gpu_d2h_chunk_size` | Maximum events stored in one pinned-host D2H buffer; `0` disables automatic D2H |
| `gpu_memory_budget_gb` | Per-BD committed device-memory limit |
| `batch_size` | EventBuilder communication size, not an execution-slot capacity |

`gpu_d2h_chunk_size` does **not** determine the number of GPU slots. For
example, `n_gpu_streams=2, gpu_d2h_chunk_size=2` means two reusable execution
slots and pinned D2H buffers that can hold up to two event results each.

The `ds_count_events.py` option `--gpu_pool_depth` is an alias for
`n_gpu_streams`. Its `--gpu_d2h_interval` option is a benchmark-consumer
control: it calls `on_cpu()` once every N events and is not part of the
DataSource scheduling policy.

---

## Ownership objects

### `_EventSlot`: one occupied execution pipeline

`EventPool.submit()` creates an `_EventSlot` record containing:

- The reusable slot ID and its producer CUDA stream.
- GPU results indexed by event timestamp.
- CPU event objects for delivery.
- One `SlotLease` per exposed result.
- Pending pinned-host D2H tokens.
- Independent CPU fallback results.

The record remains owned by `EventPool` until final retirement. Result routing
is enqueued on the producer stream before the shared result-ready event is
recorded.

### `SlotLease`: terminal-consumer completion

Each result lease contains:

- `result_ready`: the producer/result-ready CUDA event.
- `_consumer_done`: the terminal-consumer CUDA event, when one is registered.

Automatic D2H registers its copy-completion event. An external
`on_gpu_view(stream)` consumer registers an event recorded after the user's
kernel launches. `EventPool` waits for the registered event before reuse.

### `_PinnedSlot`: bounded host retention

Each automatic-D2H pipeline lazily allocates a bounded pool of page-locked
host buffers. A pinned slot contains:

- A NumPy view over CUDA pinned memory.
- One D2H completion event.
- A reference count for the event results stored in that buffer.
- A reference to the pipeline's free-slot queue.

The number of pinned slots is `max(2, n_gpu_streams)`. Each pinned slot can
hold up to `gpu_d2h_chunk_size` event results.

### `_PendingD2H`: one event's pinned result

The token stores a pinned slot, row index, and segment count. Calling
`_PendingD2H.get()`:

1. Synchronizes the pinned slot's D2H completion event.
2. Copies the event row into an independent pageable NumPy array.
3. Decrements the pinned-slot reference count.
4. Returns the pinned slot to the free queue when its last result is consumed.

If the user never calls `on_cpu()`, token destruction releases the pinned-slot
reference. The execution slot and pinned-host slot therefore have separate
lifetimes.

---

## Producer and automatic-D2H scheduling

Automatic D2H is armed when GPU work is submitted, not when the result is
later yielded:

```text
producer stream:
    H2D input is ready
      -> calibration and final routing
      -> record result-ready event

D2H stream:
    wait for result-ready event
      -> cudaMemcpyAsync(slot result -> pinned host)
      -> record D2H-done event
      -> register D2H-done on SlotLease
```

`_D2hPipeline.schedule(slot_record)` schedules every matching result in the
submitted execution slot. It divides the record into physical chunks no larger
than `gpu_d2h_chunk_size`. A partial chunk is scheduled immediately; there is
no cross-slot `_chunk_buf` and no batch-boundary D2H flush.

For `batch_size=1, gpu_d2h_chunk_size=2`, each execution record contains one
event, so a one-event partial D2H is issued for every event.

---

## Two retirement policies

### Automatic D2H: `gpu_d2h_chunk_size > 0`

Automatic D2H is the CPU-result mode. The normal slot-replacement path is:

```text
submit event N in slot S
  -> calibration/routing
  -> schedule D2H immediately
  -> D2H completes into pinned host memory

when slot S is needed for event N + pool_depth:
  begin_retire_next()
  -> verify every exposed result has a pending D2H token
     or an independent CPU fallback
  -> finish_retire_next() waits for D2H-done
  -> release slot S
  -> issue replacement H2D into slot S
  -> yield event N as a host-backed context
```

If any result key lacks a host handoff, the controller conservatively keeps
the yield-first two-phase retirement path and does not release the device slot
before delivery.

The replacement H2D is issued before the outgoing event is delivered to user
CPU code. This permits work in another execution slot to overlap with that H2D.

The normally retired context is explicitly marked `device_released`:

- Result keys remain available through `ctx.get()`.
- `on_cpu` consumes the pending pinned token or cached CPU result.
- The context does not retain the old slot-backed CuPy array or its lease.
- `on_gpu` and `on_gpu_view` raise a clear error.

This prevents a host-backed event from exposing a stale GPU view after its
device slot has been reused.

At EndRun, BeginStep, end-of-input, or `max_events`, `EventPool.flush()` drains
remaining records without scheduling a replacement H2D. It still preserves
the consumer-registration and completion ordering before clearing each slot.

### External GPU consumer: `gpu_d2h_chunk_size = 0`

This mode preserves the slot-backed result while user code registers a GPU
consumer:

```text
begin_retire_next()
  -> synchronize producer
  -> mark slot RETIRING but keep it occupied
  -> yield slot-backed GPU result
  -> user launches downstream kernel
  -> on_gpu_view().__exit__ records and registers consumer-done event
  -> finish_retire_next() waits for consumer-done
  -> release slot
  -> issue replacement H2D
```

The yield between `begin_retire_next()` and `finish_retire_next()` is the
registration window. `submit()` rejects attempts to overwrite a slot while
retirement is incomplete.

External slot-backed results must be consumed during the current generator
iteration, before requesting the next event. Retaining a slot-backed context
and first accessing its GPU array after generator advancement is not safe.

---

## Why two-phase external retirement is required

The old order inspected the leases and freed the slot before yielding the
result:

```text
retire slot 0
  -> no external consumer registered yet
  -> free slot 0
issue event 1 input
yield event 0
user launches delayed kernel for event 0 and registers completion too late
event 1 calibration overwrites slot 0
```

The corrected order retains the slot while the user registers work:

```text
begin retirement of slot 0
  -> keep slot 0 occupied
yield event 0
user launches kernel and registers consumer-done
finish retirement
  -> wait for consumer-done
  -> free slot 0
issue and submit event 1
```

`gpu_slot_overwrite_repro.py` exercises this race using `pool_depth=1`,
`batch_size=1`, automatic D2H disabled, and a delayed external kernel. On the
fixed implementation it should be run with `--expect safe`.

---

## `GPUResult` access modes

### `on_cpu`

`on_cpu` has three ordered paths:

```python
if self._cpu_cache is not None:
    return self._cpu_cache

if self._pending_d2h is not None:
    self._cpu_cache = self._pending_d2h.get()
    self._pending_d2h = None
    return self._cpu_cache

if self._device_released or self._arr is None:
    raise RuntimeError("host handoff is incomplete")

self._cpu_cache = self._arr.get()
return self._cpu_cache
```

The paths are:

1. Return an independent cached CPU result.
2. Consume the automatic pinned D2H token, copy to independent CPU memory,
   cache it, and return it.
3. With automatic D2H disabled, perform one blocking `arr.get()`, cache it,
   and return it.

Caching path 3 is required because a second D2H from the same slot-backed
array could otherwise read a newer event after slot reuse.

### `on_gpu`

With automatic D2H disabled, `on_gpu` returns an independent D2D copy:

```python
arr = ctx.get("jungfrau.calib").on_gpu
```

The copy should be requested during the current event iteration on the CuPy
null/default stream. The next slot submission synchronizes that stream before
overwriting the calibration output slot. For a custom consumer stream, use
`on_gpu_view(stream)` so its completion event is registered explicitly.

In the normal automatic-D2H delivery path, device storage has already been
released and `on_gpu` raises. Select `gpu_d2h_chunk_size=0` when the result is
intended for GPU consumption.

### `on_gpu_view(stream)`

This is the explicit zero-copy external-consumer API:

```python
stream = cp.cuda.Stream(non_blocking=True)

for ctx in run.events():
    with ctx.get("jungfrau.calib").on_gpu_view(stream) as arr:
        downstream_kernel(arr, stream=stream)
```

All kernels reading the view must be enqueued on the supplied stream inside
the `with` block. `__exit__` records a done-event on that stream and registers
it with the result lease.

Automatic D2H and `on_gpu_view()` are intentionally mutually exclusive. Use
`gpu_d2h_chunk_size=0` for this path.

---

## Pinned-host backpressure and fallback

Pinned buffers remain claimed until their result tokens are consumed or
destroyed. If every pinned slot is retained, `_D2hPipeline.schedule()` does
not defer an unsafe fallback to a future `on_cpu()` call.

Instead, while the device lease is still valid, it:

1. Synchronizes the result-ready event.
2. Calls blocking `arr.get()`.
3. Stores the independent NumPy result in
   `_EventSlot.cached_cpu_results_by_ts`.
4. Allows the execution slot to retire safely.

This keeps memory bounded and avoids both deadlock and delayed reads from a
reused device slot. Automatic D2H resumes when a pinned slot returns to the
free queue.

The pinned-slot reference counter is protected by a lock so concurrent
`on_cpu()` calls cannot lose a decrement and permanently remove a slot from
the free queue.

---

## Backpressure layers

The implementation has three distinct bounds.

### 1. Execution-slot backpressure

`EventPool` has `n_gpu_streams` reusable slots. A requested slot cannot be
reused until its producer and terminal-consumer events have completed.

```text
free execution slot exists -> admit work
all required slots occupied -> CPU scheduler waits at retirement
terminal consumer completes -> slot credit returns
```

### 2. Pinned-host backpressure

The automatic-D2H pinned pool is bounded by its slot count and chunk capacity.
When it is exhausted, results are materialized synchronously into independent
CPU memory rather than allocating without limit or exposing a stale device
view.

### 3. Device-memory backpressure

`_GpuBudget` accounts for committed device allocations. One coherent EB batch
is divided into byte-bounded `GpuSubbatchView` objects before submission.

KvikIO raw input slots and calibrated output slots reserve committed bytes
directly. Fixed constants and geometry are subtracted when deriving the
per-subbatch allowance. Detector raw/gather scratch and input bytes are
included in subbatch estimation, while the remaining CuPy allocator overhead
is covered by the safety margin.

Pinned host memory is tracked separately through `_D2hPipeline.pinned_bytes()`.

These controls are related but not interchangeable: pool depth bounds active
execution pipelines, D2H chunk size bounds each host-transfer buffer, and the
GPU budget bounds committed VRAM bytes.

---

## Pool-depth concurrency

Within one execution slot, reuse follows a protected ownership chain:

```text
raw input -> parsed raw -> calibrated result -> terminal consumer -> release
```

For automatic D2H, D2H is the terminal device consumer. For external mode,
the user-registered kernel is the terminal consumer.

With `pool_depth=1`, the same slot is required for every event, so the
pipeline is effectively sequential:

```text
slot 0: H2D0 -> calib0 -> D2H0 -> release -> H2D1 -> calib1 -> ...
```

With `pool_depth=2`, two protected pipelines can overlap:

```text
slot 0: H2D0 -> calib0 -> D2H0 -> release -> H2D2 -> ...
slot 1: H2D1 -> calib1 -> D2H1 -> release -> H2D3 -> ...
```

For the normal automatic-D2H replacement path, the controller releases the
outgoing slot and issues its replacement H2D before yielding the host-backed
event. This permits, for example, H2D2 to overlap D2H1 when timing allows.

In a ten-event Nsight Systems run with `batch_size=1`, `pool_depth=2`, and
`gpu_d2h_chunk_size=2`, 8 of the 9 possible D2H-to-next-H2D pairs overlapped.
This is a measured example, not a performance assertion in the unit tests.

---

## KvikIO read ownership and early termination

The controller permits at most one KvikIO `PendingBatch` to be pre-issued
ahead of CPU event processing. Because automatic mode can issue replacement
H2D before yielding a CPU event, the generator may be closed while that read
is still in flight.

`GpuEvents` therefore:

- Stores the owned pending read in `_pending_gpu_read`.
- Rejects a second pre-issued read while one is outstanding.
- Clears ownership when `wait_batch()` completes.
- Drains an outstanding read before `gpu_reader.close()` during shutdown.

This prevents reader buffers and file state from being closed underneath an
in-flight H2D.

---

## Transition and drain behavior

BeginStep drains dependent GPU work before calibration constants are replaced.
EndRun, end-of-input, and `max_events` drain occupied slots exactly once.

`EventPool.flush()`:

1. Synchronizes the producer stream.
2. Yields the record while retaining slot ownership.
3. Allows a terminal consumer to register.
4. Waits for registered consumer events in `finally`.
5. Clears the slot.

The `finally` block also protects generator close and exceptions during result
delivery.

---

## Usage rules

| Configuration and access | Safe use |
|---|---|
| Automatic D2H plus `on_cpu` | Supported; contexts may retain their host token or cached CPU result |
| Automatic D2H plus `on_gpu_view` | Unsupported; use `gpu_d2h_chunk_size=0` |
| External mode plus `on_gpu` | Request the independent copy on the null/default stream during the current iteration |
| External mode plus `on_gpu_view(stream)` | Enqueue all readers on `stream` inside the `with` block during the current iteration |
| External mode plus `on_cpu` | First call blocks and caches; call before advancing if the result is still slot-backed |
| Retain an unconsumed external slot-backed context | Unsafe; its slot may be reused after generator advancement |
| Ignore automatic-D2H results | Safe; token destruction releases pinned references, but the D2H bandwidth is wasted |

Do not infer safety from Python object lifetime alone. A slot-backed GPU view
must have an explicit CUDA completion event registered before the generator
advances.

---

## Validation

The GPU unit tests cover:

- A slot cannot be submitted before retirement finishes.
- A consumer registered after `begin_retire_next()` is observed by
  `finish_retire_next()`.
- Generator advancement alone does not complete a lease.
- The D2H stream waits for the result-ready event.
- Complete and partial execution-slot records receive D2H tokens immediately.
- `on_cpu()` returns correct data and caches synchronous fallback results.
- Pinned exhaustion materializes a safe CPU fallback before device reuse.
- Concurrent pinned-slot reference decrements return the slot exactly once.
- `on_gpu()` returns an independent copy.
- `on_gpu_view()` records the supplied stream's completion event.
- Byte-bounded subbatch and `_GpuBudget` invariants.

The overwrite diagnostic is:

```bash
mpirun -n 3 \
  python -u psana/psana/debugtools/gpu_slot_overwrite_repro.py \
    --exp mfx100848724 \
    --run 51 \
    --dir /sdf/data/lcls/ds/prj/public01/xtc \
    --expect safe \
    --log-level WARNING
```

The timeline driver is `psana/psana/debugtools/ds_count_events.py`. With
`--gpu_d2h_interval 1`, it calls `on_cpu()` for every event so automatic D2H
and its CPU delivery are visible in Nsight Systems.

---

## Current source map

| File | Responsibility |
|---|---|
| `gpu_events.py` | Event controller, automatic D2H, host tokens, retirement policy, KvikIO pending-read ownership |
| `gpu_stream.py` | `_EventSlot`, `EventPool`, slot leases, two-phase retirement |
| `context.py` | `GPUResult`, `on_cpu`, `on_gpu`, `on_gpu_view`, host-only result state |
| `gpu_kvikio_read.py` | Reusable per-slot raw input buffers and asynchronous reads |
| `gpu_calib.py` | Per-slot calibration/raw buffers and result-ready producer work |
| `gpu_budget.py` | Committed device-memory accounting |
| `gpu_batch.py` | GPU batch and byte-bounded subbatch views |

---

## Deferred work

- Aggregate GPU-memory coordination across multiple BD processes sharing one
  device remains separate from the per-BD budget.
- True GDS depends on filesystem and cuFile infrastructure. On Lustre/GPFS,
  KvikIO uses the CPU-fallback path (storage to CPU DRAM to GPU VRAM).
- Logical joins of many CPU results are separate from physical D2H chunking.
  Compact downstream GPU reductions should transfer only their reduced result
  rather than full calibrated detector planes.
