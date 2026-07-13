# GPU Memory Backpressure and Asynchronous D2H Join Design

## Status

Proposed design, not yet implemented.

This design applies to the integrated GPU path at commit `6f0ef8b93`:

```text
GpuEvents
  -> KvikioGpuReader
  -> EventPool
  -> GPUDetector.process_batch()
  -> fused_calib_gpu()
  -> GpuEventContext / GPUResult
```

Backpressure work should modify this path only. It must not restore the deleted
standalone pipeline, `StreamPool`, `GPUKernelRegistry`, or obsolete MPI helper
functions.

## Goal

Provide bounded GPU and pinned-host memory use while preserving overlap among:

```text
bigdata read / CPU fallback H2D
GPU raw gather and calibration
downstream GPU processing
asynchronous D2H
CPU/GPU result joining
```

The central lifetime rule is:

> A GPU slot may be reused only after every consumer of that slot has completed.

Advancing the Python event generator is not proof that an asynchronous GPU or
D2H consumer has completed.

## Performance Evidence

See [d2h_interval_bandwidth_results.md](d2h_interval_bandwidth_results.md).

The original measurements used one BD and one GPU with KvikIO CPU fallback:

| Run | Loop time | Rate | D2H calls | D2H time |
| --- | ---: | ---: | ---: | ---: |
| CPU path | 419.61 s | 38.1 Hz | 0 | 0 |
| GPU, no D2H | 159.47 s | 100.3 Hz | 0 | 0 |
| GPU, D2H interval 100 | 162.10 s | 98.7 Hz | 160 | 2.63 s |
| GPU, D2H interval 10 | 185.39 s | 86.3 Hz | 1600 | 25.90 s |
| GPU, D2H every event | 404.16 s | 39.6 Hz | 16000 | 246.37 s |

`gpu_d2h_interval=N` is a sampling knob. It copies one event every N events; it
does not copy all N events in one operation. These results show that synchronous
per-event D2H is additive and that infrequent D2H preserves GPU throughput. They
motivate an asynchronous joined path, but do not prove that retaining 100 full
images or transferring them as one allocation is efficient.

## Current Limitation

The current `EventPool` bounds the number of in-flight batches with
`n_gpu_streams`. It does not bound total bytes.

One full EventBuilder GPU batch maps to one EventPool slot. Before recycling a
slot, EventPool synchronizes its CUDA stream and yields the old
`GpuEventContext` objects. When iteration resumes, the slot may be overwritten
by a later batch. No D2H occurs merely because a context is yielded.

Current reusable or persistent GPU allocations include:

- One KvikIO raw-input buffer per slot.
- One calibrated-output buffer per slot.
- Raw-gather buffers per slot and stream layout.
- Per-stream reordered pedestal and gain/mask caches.
- Calibration constants and optional geometry arrays.
- CuPy memory-pool cached blocks.

The slot buffers grow to their observed high-water sizes and generally do not
shrink. `n_gpu_streams=2` therefore means at most two batches, not at most a
specific number of GiB. `free_calib_bufs()` drops only calibrated-output
references and is not a complete backpressure mechanism.

## Separate The Control Units

Three sizes have different purposes and must remain independent:

```text
EB batch_size       Number of events communicated from EB to a BD
GPU subbatch        Byte-bounded work assigned to one execution slot
join_size           Logical number of CPU-visible results delivered together
```

Two additional D2H controls are required:

```text
d2h_chunk_size      Maximum events or bytes in one physical D2H operation
d2h_max_inflight    Maximum simultaneous D2H operations
```

For example:

```text
join_size=100
d2h_chunk_size=10
```

The implementation may transfer ten results at a time, release the associated
GPU slots, accumulate the completed chunks on the host, and deliver one logical
100-event join.

## Memory Budgets

### GPU budget

Each BD needs an explicit quota when multiple BDs share one GPU. The first
prototype can use static per-BD quotas. A later GPU-wide coordinator can provide
dynamic fairness and account for allocations shared among processes.

The budget must include all owners, not only calibrated output:

```text
M_gpu =
    M_fixed_constants_caches_geometry
  + sum(M_raw_input_slot_capacity)
  + sum(M_gather_scratch_slot_capacity)
  + sum(M_output_slot_capacity)
  + M_retained_gpu_results
  + M_allocator_margin
```

The invariant is:

```text
M_gpu <= per_bd_gpu_budget
```

For a 40 GiB A100 shared by two BDs, an initial policy might reserve 10 GiB for
fixed state, allocator behavior, and safety margin, then assign approximately
15 GiB to each BD. These values must be measured and configurable, not embedded
as Jungfrau constants.

### Capacity versus active ownership

Reusable buffers remain allocated after a slot completes. The controller must
track two related quantities:

```text
committed capacity  Physical reusable allocations currently owned by the BD
active leases       Which committed buffers cannot currently be overwritten
```

Completing a lease makes capacity reusable; it does not necessarily return VRAM
to CUDA. Before growing a reusable buffer, the controller must reserve the
additional bytes and reject growth that would exceed the quota.

### Pinned-host budget

Pinned host memory needs a separate limit:

```text
M_pinned = sum(inflight_d2h_chunk_capacity) <= pinned_host_budget
```

Unbounded pinned allocations can damage the node even when GPU memory remains
within quota. `d2h_max_inflight=2` is useful only when two chunks fit both the
GPU and pinned-host budgets.

## Byte-Bounded GPU Subbatches

EventBuilder `batch_size` remains the communication unit. `GpuEvents` partitions
the received event and descriptor tables into GPU execution subbatches.

The estimator should account for each event's:

```text
bigdata input bytes
raw-gather bytes
calibrated-output bytes
detector-specific workspace
requested downstream-result bytes
```

Partitioning should use estimated bytes or work, not equal event counts, because
events may have different descriptor counts, missing streams, or payload sizes.

One EB batch can populate several execution slots:

```text
EB batch: events 0..19
pool_depth: 2

slot 0 <- first byte-bounded subbatch
slot 1 <- next byte-bounded subbatch
remaining subbatches wait in a bounded local queue
```

This separates communication efficiency from GPU concurrency and avoids
requiring two EB round trips merely to fill two CUDA streams.

If a batch does not fit:

1. Split it into smaller GPU subbatches.
2. Admit only the number of subbatches for which memory credits and slots exist.
3. Temporarily run fewer active slots if the budget requires it.
4. If one event cannot fit even with exclusive use of the BD quota, raise a
   specific memory-pressure error or use a documented fallback.

Do not silently change the user-visible EB `batch_size` or promise more
concurrency than the budget allows.

## Slot Leases And Completion Events

Each execution slot should follow an explicit state machine:

```text
FREE
  -> READING
  -> COMPUTING
  -> RESULT_READY
  -> GPU_CONSUMER or D2H_IN_FLIGHT
  -> FREE
```

Relevant CUDA completion points include:

```text
read_done
calib_done
downstream_done
d2h_done
```

The final consumer returns an event or completion token. EventPool may recycle
the slot only after that token completes.

For an asynchronous D2H on a separate stream:

```text
calibration stream:  ... -> record calib_done
copy stream:         wait calib_done -> D2H -> record d2h_done
EventPool:           wait/query d2h_done before slot reuse
```

A `GPUResult` used directly by downstream GPU code also needs explicit lifetime
semantics, conceptually:

```python
arr = result.on_gpu
result.release_after(downstream_done)
```

If a user retains a lease without releasing it, the bounded scheduler should
stop admitting work rather than overwrite the array. Framework-managed joins
should normally consume and release leases automatically so a sequential event
loop cannot deadlock while waiting to reach a future join checkpoint.

## Asynchronous D2H Join

### Logical join versus physical transfer

`join_size` is a logical delivery target. It should not require that all joined
full-size results remain in VRAM simultaneously.

The physical D2H chunk size is selected from available GPU and pinned-host
credits. Completed chunks can be accumulated in normal host-owned result
storage until `join_size` events are available.

The joiner must preserve timestamp and event-index metadata so CPU and GPU
results are matched deterministically even when subbatches or D2H operations
complete out of order.

### Full calibrated output

For full images, copy directly from the execution-slot output into a bounded
pinned-host chunk:

```text
calibrated slot view
  -> async D2H to pinned host chunk
  -> d2h_done
  -> release execution-slot lease
```

Do not first copy every image into a second full-size GPU join buffer. That D2D
step duplicates VRAM and consumes HBM bandwidth without improving ownership.

### Compact downstream output

The likely production result is smaller than the calibrated image, for example
angular-integration bins. Prefer:

```text
raw input
  -> calibration
  -> GPU reduction/binning
  -> compact result
  -> async D2H
  -> release full-image execution slot
```

If the compact result is small, kernels may write directly into a bounded GPU
join accumulator. A single D2H for `join_size` compact results can then be both
safe and efficient.

### Partial joins

The implementation must flush fewer than `join_size` events on:

- BeginStep before calibration constants are replaced.
- EndRun.
- End of input or `max_events`.
- Explicit user flush.
- Host-memory pressure that prevents accumulating the logical join.

GPU-memory pressure alone should normally cause an earlier physical D2H chunk,
not necessarily a smaller logical join. If host storage also cannot retain the
requested join, return a partial join rather than deadlock.

## Example Timeline

For 20 events with `batch_size=4`, `pool_depth=2`, and `join_size=10`:

```text
EB batch 0: events 0..3
  slot 0: events 0..1 -> compute -> async D2H chunk
  slot 1: events 2..3 -> compute -> async D2H chunk
  release each slot after its d2h_done event

EB batch 1: events 4..7
  reuse the first completed slots
  append host results 4..7

EB batch 2: events 8..11
  process events 8..9
  deliver logical join 0..9
  events 10..11 begin the next logical join

Continue through event 19 and deliver join 10..19.
```

At most the admitted subbatches and bounded D2H chunks occupy GPU memory. The
execution slots do not remain leased for ten events merely because
`join_size=10`.

## Backpressure Propagation

When no slot and memory credits are available:

1. Stop admitting GPU subbatches.
2. Stop receiving/requesting additional EB batches when the bounded pending
   queue is full.
3. Poll or wait for a downstream or D2H completion event.
4. Release the completed lease and mark its capacity reusable.
5. Admit the next queued subbatch.

This propagates pressure toward EB without allocating an unbounded local queue.
The queue limit should be expressed in bytes as well as batch count.

For multiple BDs sharing one GPU, static per-BD quotas are the simplest first
implementation. A production implementation needs GPU-wide accounting and
fairness so all BD processes cannot simultaneously grow to the device limit.

## Transition Handling

The existing correctness rules remain mandatory:

- BeginStep drains work that depends on the old calibration constants before
  those constants are updated.
- Ordinary transitions do not drain unrelated GPU work.
- EndRun drains pending subbatches and D2H operations exactly once.
- A partial logical join is emitted at a required drain boundary.

The segment-ordering fix must also remain intact: calibration constants follow
the child-XTC segment order observed in L1Accept, not merely sorted Configure
segment IDs.

## Proposed Internal Responsibilities

No new public API is required for the first prototype. Responsibilities should
remain in the current pipeline:

### `GpuEvents`

- Orchestrate batch partitioning, admission, CPU/GPU timestamp joining, and
  transition drains.
- Bound the queued EB and GPU subbatch bytes.
- Drive the optional asynchronous D2H join consumer.

### `KvikioGpuReader`

- Report required input-buffer growth before allocation.
- Reserve memory credits before growing a slot buffer.
- Associate pending reads with the execution-slot lease.

### `EventPool`

- Own slot states, CUDA streams, leases, and final-consumer completion tokens.
- Never recycle a slot based only on Python generator progress.
- Admit only work for which a free slot and byte credits exist.

### `GPUDetector`

- Estimate gather, output, and workspace bytes for a proposed subbatch.
- Reserve credits before growing reusable buffers.
- Preserve calibration and gather ordering on the slot stream.

### `GPUResult` / `GpuEventContext`

- Carry a result lease when device data escapes to user GPU code.
- Allow downstream code or an internal D2H consumer to attach a completion
  token.
- Make lifetime violations explicit rather than returning overwritten views.

## Suggested Configuration

Names are provisional:

```text
gpu_memory_budget_bytes
gpu_pending_queue_bytes
gpu_join_size
gpu_d2h_chunk_bytes
gpu_d2h_max_inflight
gpu_pinned_memory_budget_bytes
```

Existing controls retain their meanings:

```text
batch_size       EB communication size
n_gpu_streams    Maximum concurrent execution slots
```

The scheduler may use fewer than `n_gpu_streams` when memory credits are not
available.

## Implementation Phases

### Phase 0: measurement and accounting

- Add per-owner current and high-water byte counters.
- Report fixed state, raw input, gather, output, CuPy pool, and pinned memory.
- Validate estimates against CUDA memory information on A100.

### Phase 1: correct slot ownership

- Add slot leases and final-consumer CUDA completion tokens.
- Keep current one-EB-batch-per-slot scheduling initially.
- Prove that an in-flight D2H or downstream kernel prevents slot reuse.

### Phase 2: bounded asynchronous D2H

- Add direct async D2H from execution output to pinned chunks.
- Separate logical `join_size` from physical D2H chunk bytes.
- Support partial-tail and transition flushes.
- Avoid the old full-size D2D join buffer.

### Phase 3: byte-bounded subbatches

- Add memory estimation and reservation.
- Partition one EB GPU batch into execution subbatches.
- Fill available streams from one EB communication.
- Bound the pending queue and propagate backpressure.

### Phase 4: shared-GPU coordination

- Enforce aggregate budgets across BDs sharing one GPU.
- Add fairness and diagnostics per BD rank and GPU ID.
- Account correctly for shared calibration allocations.

## Validation

Preserve the existing baseline:

- CPU unit invariants in `tests/gpu/unit/test_core.py`.
- Pixel-exact A100 acceptance in
  `tests/gpu/integration/test_pixel_exact.py`.
- Single-event, batched slot reuse, and partial-tail coverage.

Add tests for:

- A slot cannot be recycled while D2H is in flight.
- A downstream CUDA completion token controls release.
- Generator advancement alone does not release a lease.
- Subbatch estimates and actual allocation stay within budget.
- Variable event sizes split correctly while preserving order.
- A single oversized event fails explicitly.
- BeginStep and EndRun flush partial joins correctly.
- Ordinary transitions do not force unnecessary drains.
- Multiple D2H chunks produce one correctly ordered logical join.
- Multiple BDs cannot exceed the configured aggregate GPU budget.

Performance remains script-driven rather than asserted in pytest. Measure:

```text
event rate and loop time
read/H2D, gather, kernel, and D2H overlap
D2H bytes and effective bandwidth
GPU committed and active bytes by owner
pinned-host current and high-water bytes
slot and memory-credit stall time
pending queue depth and bytes
NIC receive bandwidth
```

Compare no D2H, synchronous per-event D2H, bounded asynchronous full-result
D2H, and compact-result joins. On the S3DF CPU-fallback path, also determine how
D2H competes with storage-to-CPU-to-GPU traffic for PCIe and memory bandwidth.

## Success Criteria

The design is successful when:

1. GPU and pinned-host memory remain within configured budgets.
2. No result is overwritten before its final consumer completes.
3. Large EB batches are split without changing event order or join semantics.
4. `join_size` does not require retaining `join_size` full images in VRAM.
5. D2H overlaps useful read or compute work when hardware permits.
6. Backpressure reaches EB instead of producing unbounded local allocation.
7. Transition and pixel-exact calibration correctness remain unchanged.
