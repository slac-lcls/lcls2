# Join Operation — Combining CPU and GPU Results at Event Checkpoints

**Branch:** `features/psana2-gpu`  
**Status:** Design proposal — **NOT YET IMPLEMENTED**

> **Current constraint:** GPU results (`ctx.get('calib').on_gpu`) are views into
> pre-allocated EventPool slot buffers.  These buffers are recycled after
> `n_slots` (default 4) batches.  The user **must consume GPU results
> immediately in the loop iteration** — storing raw `on_gpu` references
> and reading them after `n_slots` more batches have been processed will
> return overwritten (stale) data.  See the "Current usage constraint" section
> below for the safe accumulation pattern.

---

## Proposed user API

```python
for i_evt, ctx in enumerate(run.events()):
    calib  = ctx.get('jungfrau.calib').on_gpu    # CuPy, on this BD rank's GPU
    energy = ctx.raw('gmd').energy               # float64, CPU scalar
    # accumulate partial result locally
    my_histogram += compute_histogram(calib, energy)

    if i_evt > 0 and i_evt % 1000 == 0:
        global_histogram = join(my_histogram, across='all_bd_ranks')
        # global_histogram combines contributions from every BD rank
```

---

## What "CPU result" and "GPU result" mean

Each BD rank already has **both** CPU and GPU detector data for its own events
via `GpuEventContext`:

| Term | Source | Type | Example |
|---|---|---|---|
| `cpu_result` | `ctx.raw('gmd').energy` | Python scalar / NumPy | Beam energy, pulse energy, transmission |
| `cpu_result` | `ctx.raw('ipm').xpos` | Python scalar / NumPy | Beam position, intensity monitor |
| `gpu_result` | `ctx.get('jungfrau.calib').on_gpu` | CuPy array (32, 512, 1024) | Calibrated Jungfrau image |
| `gpu_result` | `ctx.get('jungfrau.image').on_gpu` | CuPy array (4432, 4216) | Assembled 2-D detector image |

The user **accumulates** these separately over N events on each BD rank, then
calls `join()` to combine them across all BD ranks and compute a combined result.

### Concrete example: beam-intensity-normalised average image

```python
# Each BD rank accumulates its own events:
sum_image    = cp.zeros((32, 512, 1024), dtype=cp.float32)  # GPU accumulator
sum_energy   = 0.0                                           # CPU accumulator
n_hits       = 0

for i_evt, ctx in enumerate(run.events()):
    calib  = ctx.get('jungfrau.calib').on_gpu   # gpu_result for this event
    energy = ctx.raw('gmd').energy              # cpu_result for this event

    sum_image  += calib
    sum_energy += energy

    if i_evt > 0 and i_evt % 1000 == 0:
        # join: AllReduce both across all BD ranks, then combine them
        global_image, global_energy = join(sum_image, sum_energy)
        normalised_avg = global_image / global_energy   # beam-normalised image
        save(normalised_avg)

        sum_image.fill(0)   # reset accumulators
        sum_energy = 0.0
```

### Why joining across BD ranks is necessary

EB distributes **batches** of events to BD ranks using a first-come-first-served
policy with a soft round-robin tiebreak.  Whichever BD rank sends a request to
EB first receives the next batch.  When multiple ranks are simultaneously waiting,
EB picks the one closest to a round-robin cursor (`rr_next_bd`) among the ready
set.  Strict round-robin (waiting for a specific rank even if others are ready)
only applies when `smd_callback` is set, which is not the GPU path.

In practice with 2 BD ranks and similar processing times, the distribution
is approximately alternating: with `batch_size=80`:

- BD rank 0 tends to receive batch 0 (events 0–79), batch 2 (events 160–239), …
- BD rank 1 tends to receive batch 1 (events 80–159), batch 3 (events 240–319), …

But the exact assignment depends on which rank finishes processing first and
sends its next request.  After each rank has processed `~1000 / n_bd_ranks`
events worth of batches:
- BD rank 0 has a partial sum from its batches (≈ 500 events if n_bd_ranks=2)
- BD rank 1 has a partial sum from its batches (≈ 500 events)

Neither rank alone has the complete 1000-event statistics.  `join()` does an
MPI AllReduce across all BD ranks to produce the global sums:

```
BD rank 0: batches 0,2,4,…  →  sum_image_0 (~500 events), sum_energy_0  ─┐
                                                                             ├─ join()
BD rank 1: batches 1,3,5,…  →  sum_image_1 (~500 events), sum_energy_1  ─┘
                                    ↓
                     global_image  = sum_image_0  + sum_image_1  (≈ 1000 events)
                     global_energy = sum_energy_0 + sum_energy_1
```

---

## Design: checkpoint-based cross-rank reduction

### Concept

EB distributes batches (not individual events) to BD ranks first-come-first-served.
Each BD rank accumulates both a CPU partial result (scalars from `ctx.raw()`)
and a GPU partial result (arrays from `ctx.get()`) over its assigned batches.
When every BD rank has processed enough batches to collectively cover
`n_checkpoint` events, `join()` does:

1. **AllReduce** both partial results across all BD ranks (MPI or NCCL)
2. Return the **global** CPU and GPU results to every BD rank
3. The user combines them (normalise, correlate, etc.)

```
batch_size = 80, n_bd_ranks = 2

BD rank 0:  batches 0,2,4,… (≈500 events)  → sum_image_0, sum_energy_0  ─┐
                                                                            ├─ join()
BD rank 1:  batches 1,3,5,… (≈500 events)  → sum_image_1, sum_energy_1  ─┘
                                                                         ↓
                                global_image  = sum_image_0  + sum_image_1  (≈1000 events)
                                global_energy = sum_energy_0 + sum_energy_1
```

### Where the collective can be called

**Problem:** `bd_comm` spans EB (bd_rank=0) + BD workers (bd_rank≥1).
Any collective on `bd_comm` requires EB to participate, but EB is in
`eb_node.start()` during the event loop — calling a collective there deadlocks.

The safe communicator is a **BD-workers-only** communicator that excludes EB.
`create_gpu_communicators()` in `gpu_mpi.py` provides the building blocks:

```python
# gpu_mpi.py — called in share_calib_between_gpu_peers():
comms = create_gpu_communicators(comm, bd_ranks)
# comms.bd_comm   — BD ranks + EB (rank 0 = EB, ranks 1..N = BD workers)
#                   NOT safe for user collectives (EB is in start(), deadlocks)
# comms.node_comm — intra-node COMM_TYPE_SHARED (for GPU P2P / NCCL)
```

Note: `create_gpu_communicators()` is currently called only inside
`share_calib_between_gpu_peers()` for CUDA IPC.  It is **not** called in
`_gpu_events_mpi()`.  For `join()`, a new BD-workers-only sub-communicator
(excluding EB) would need to be created and exposed — this is step 1 of the
implementation plan below.

A `join()` call inside the BD rank's event loop would use a BD-workers-only
communicator for MPI AllReduce or `comms.node_comm` + NCCL for GPU operations.

### Two reduction backends

| Backend | Path | When to use |
|---|---|---|
| **MPI AllReduce** | GPU → CPU DRAM → MPI → CPU DRAM → GPU | Small arrays, heterogeneous CPU/GPU results, Lustre filesystem (no GPUDirect MPI) |
| **NCCL AllReduce** | GPU VRAM → (NVLink/PCIe) → GPU VRAM | Large GPU arrays, homogeneous result types, `comms.bd_comm` available as NCCL communicator |

S3DF does not yet have NCCL communicators wired into psana2; the MPI path is
the immediate option.

---

## `join()` API sketch

```python
def join(cpu_result, gpu_result, op='sum', bd_comm=None):
    """Combine CPU and GPU partial results across all BD ranks at a checkpoint.

    Each BD rank has accumulated:
      cpu_result — partial result from CPU detectors (ctx.raw()), e.g.
                   sum of GMD energies, IPM positions, hit counts.
                   NumPy array, scalar, or Python number.
      gpu_result — partial result from GPU detector (ctx.get()), e.g.
                   sum of calibrated Jungfrau images.
                   CuPy array.

    join() AllReduces both across all BD workers and returns the global
    (combined) cpu_result and gpu_result.  The user then computes the
    final combined quantity (e.g. normalised_image = global_image / global_energy).

    Parameters
    ----------
    cpu_result : np.ndarray, float, or int
        Partial CPU-detector accumulator on this BD rank.
        Must have the same shape and dtype on every BD rank.
    gpu_result : cp.ndarray
        Partial GPU-detector accumulator on this BD rank (CuPy).
        Must have the same shape and dtype on every BD rank.
    op : str
        Reduction operation applied to BOTH results:
        'sum'  — element-wise sum (default; histograms, image sums, counts)
        'max'  — element-wise maximum
        'mean' — element-wise mean across all BD ranks
    bd_comm : mpi4py.MPI.Comm or None
        BD-workers-only communicator (excludes EB to avoid deadlock).
        If None, retrieved from the current run context.

    Returns
    -------
    (global_cpu_result, global_gpu_result)
        global_cpu_result — same type as cpu_result, combined across ranks
        global_gpu_result — CuPy array, combined across ranks (stays on GPU)

    Notes
    -----
    * ALL BD ranks must call join() at the same logical event-loop point.
      Mismatched calls deadlock.
    * join() is blocking — all BD ranks synchronise here.
    * gpu_result AllReduce uses MPI (D→H→AllReduce→H→D, ~26 ms for 64 MB).
      NCCL AllReduce (GPU-direct, faster) is not yet implemented.
    """
    import numpy as np
    from mpi4py import MPI

    mpi_op  = {'sum': MPI.SUM, 'max': MPI.MAX}.get(op, MPI.SUM)
    n_ranks = bd_comm.Get_size()

    # ── CPU result: straight MPI AllReduce ──────────────────────────────────
    cpu_arr     = np.asarray(cpu_result)
    global_cpu  = np.empty_like(cpu_arr)
    bd_comm.Allreduce(cpu_arr, global_cpu, op=mpi_op)
    if op == 'mean':
        global_cpu = global_cpu / n_ranks
    if np.ndim(cpu_result) == 0:
        global_cpu = global_cpu.item()         # scalar → scalar

    # ── GPU result: D→H → MPI AllReduce → H→D ───────────────────────────────
    import cupy as cp
    gpu_arr_cpu  = gpu_result.get()            # D→H  (~13 ms for 64 MB)
    global_gpu_c = np.empty_like(gpu_arr_cpu)
    bd_comm.Allreduce(gpu_arr_cpu, global_gpu_c, op=mpi_op)
    if op == 'mean':
        global_gpu_c = global_gpu_c / n_ranks
    global_gpu = cp.asarray(global_gpu_c)      # H→D  (~13 ms for 64 MB)

    return global_cpu, global_gpu
```

### Usage example — beam-intensity-normalised average image

```python
import cupy as cp
import psana

ds = psana.DataSource(exp='mfx100852324', run=77, gpu_det='jungfrau')

for run in ds.runs():
    # Per-rank partial accumulators.
    sum_image  = cp.zeros((32, 512, 1024), dtype=cp.float32)  # GPU
    sum_energy = 0.0                                           # CPU
    n_events   = 0

    for i_evt, ctx in enumerate(run.events()):
        calib  = ctx.get('jungfrau.calib').on_gpu   # gpu_result this event
        energy = ctx.raw('gmd').energy               # cpu_result this event

        sum_image  += calib
        sum_energy += energy
        n_events   += 1

        if n_events % 1000 == 0:
            # AllReduce both across all BD ranks.
            global_energy, global_image = join(sum_energy, sum_image,
                                               op='sum', bd_comm=run.bd_workers_comm)
            # run.bd_workers_comm = BD-workers-only comm (EB excluded)
            # Combine: beam-intensity-normalised average Jungfrau image.
            if global_energy > 0:
                normalised_avg = global_image / global_energy
                save_result(normalised_avg, n_total=1000 * n_bd_ranks)

            # Reset partial accumulators.
            sum_image.fill(0)
            sum_energy = 0.0
            n_events   = 0
```

### Multiple CPU and GPU accumulators

For XPCS (X-ray Photon Correlation Spectroscopy), `join()` might combine
a partial two-time correlation function (GPU) with partial beam-intensity
corrections (CPU):

```python
# Each BD rank computes partial g2 and partial I(t) over its events.
partial_g2 = cp.zeros((n_q, n_tau), dtype=cp.float64)   # GPU
partial_It = np.zeros(n_frames, dtype=np.float64)        # CPU

for i_evt, ctx in enumerate(run.events()):
    calib = ctx.get('calib').on_gpu
    I_t   = ctx.raw('ipm').sum    # beam intensity this shot

    # Update partial g2 on GPU, partial I(t) on CPU.
    update_g2(partial_g2, calib, i_evt)
    partial_It[i_evt % n_frames] = I_t

    if i_evt % 1000 == 0:
        global_It, global_g2 = join(partial_It, partial_g2)
        g2_normalised = global_g2 / normalise_by_intensity(global_It)
        save_g2(g2_normalised)
```

---

## Current usage constraint — GPU results must be consumed immediately

### Why the constraint exists

`ctx.get('calib').on_gpu` returns a **view into a pre-allocated EventPool slot
buffer**, not an independent CuPy array.  With `EventPool(n=4)` there are 4 slot
buffers, one per in-flight batch.  After 4 more batches have been submitted, slot
N is recycled and its buffer is **overwritten with new data**:

```
batch 0 → slot 0 (calib_slot_0)   ← slot 0 available to user
batch 1 → slot 1 (calib_slot_1)
batch 2 → slot 2 (calib_slot_2)
batch 3 → slot 3 (calib_slot_3)
batch 4 → slot 0 recycled          ← calib_slot_0 is OVERWRITTEN with batch 4's data
```

Any `on_gpu` array held by the user from batch 0 is now pointing to batch 4's
calibrated pixels — **silent data corruption, no error raised**.

### What is safe and what is not

```python
# ✓ SAFE: consume (accumulate) immediately in the same loop iteration.
#   sum_image += calib executes a GPU kernel that READS calib and WRITES
#   to sum_image (a separate array).  After this line, the slot can be recycled.
sum_image += ctx.get('calib').on_gpu

# ✓ SAFE: D→H copy (on_cpu) is always an independent NumPy array.
calib_cpu = ctx.get('calib').on_cpu      # D→H, independent copy, safe to store

# ✗ NOT SAFE: storing the on_gpu view and reading it later.
calib_ref = ctx.get('calib').on_gpu      # view into slot buffer
...                                       # process n_slots more batches
do_something(calib_ref)                  # slot recycled — calib_ref is stale!

# ✗ NOT SAFE: building a list of on_gpu views.
calib_list = [ctx.get('calib').on_gpu for ctx in run.events()]
# After n_slots events, early entries in calib_list point to recycled buffers.
```

### How the join accumulation pattern works safely

The `join()` accumulation pattern (`sum_image += calib`) is safe because `+=`
is an in-place kernel that reads from the slot buffer and writes to the user's
own `sum_image` array immediately.  By the time `join()` is called, all
individual `calib` views have been consumed — `sum_image` holds the accumulated
values independently of the slot buffers:

```python
sum_image = cp.zeros((32, 512, 1024), dtype=cp.float32)  # user's own array

for i_evt, ctx in enumerate(run.events()):
    calib = ctx.get('calib').on_gpu      # view into slot buffer
    sum_image += calib                   # ← GPU kernel reads slot NOW; safe
                                         # calib view is no longer needed after this line
    if i_evt % 1000 == 0:
        _, global_image = join(0.0, sum_image)   # AllReduce sum_image (independent)
        sum_image.fill(0)
```

### To store GPU results beyond the current iteration

Call `.on_cpu` to get an independent NumPy copy, or allocate a separate CuPy
array and copy:

```python
# Option 1: D→H immediately (safe to store across any number of events)
calib_saved = ctx.get('calib').on_cpu     # NumPy, independent

# Option 2: Copy to separate GPU array (stays on GPU, independent)
calib_saved = ctx.get('calib').on_gpu.copy()   # CuPy, independent

# Option 3: call free_calib_bufs() to release slot buffers entirely,
#            which falls back to dynamic allocation (no recycling).
gpu_events_obj.free_calib_bufs()
```

---

## Issues and clarifications needed

### 1. Synchronisation semantics — when do ranks reach the checkpoint?

**Problem:** EB sends each batch to whichever BD rank requests first
(first-come-first-served with soft round-robin tiebreak).  With similar
processing times, the two ranks alternate batches, but the assignment is
not guaranteed.  If each rank calls `join()` after its own 1000th event
(i.e. after processing ~1000 / batch_size batches), they reach the
checkpoint at roughly the same wall-clock time.  But if one rank is
slower (e.g., a BeginStep reconfiguration), it may have processed fewer
batches and will block the other rank at the collective.

**Clarification needed:** Does "1000 events" mean:
- **Local count:** each rank calls `join()` after processing its own 1000th
  event, regardless of other ranks' progress.  Simple, but a slow rank
  blocks faster ones at the collective.
- **Global count:** `join()` is called when the TOTAL across all ranks
  reaches 1000.  Requires an additional AllReduce to check the total first.

**Recommendation:** Local count is simpler and matches the natural
`i_evt % 1000 == 0` pattern.

### 2. `bd_comm` deadlock risk

`bd_comm` includes EB (bd_rank=0) which is in `eb_node.start()` during
the event loop.  Any collective on `bd_comm` involving EB deadlocks.

The `join()` collective must use a **BD-workers-only communicator** that
was built with `MPI_Comm_create_group` (non-collective for EB).
`create_gpu_communicators()` already creates this; the challenge is
making it accessible inside the user's event loop.

**Options:**
1. Expose `run.bd_workers_comm` as a public attribute set during `RunParallel.__init__()`.
2. Pass `bd_comm` explicitly to `join()`.
3. Store in `ctx` and provide `ctx.join(partial, op='sum')` as a method.

**Option 3** (`ctx.join()`) is cleanest from a user perspective: the context
already knows its BD communicator.

### 3. CPU vs GPU result type handling

`join()` receives either a CuPy array (GPU) or NumPy array/scalar (CPU).
The current sketch does D→H → MPI AllReduce → H→D for GPU arrays.
For large arrays (32 × 512 × 1024 × 4 bytes = 64 MB), this costs:
- D→H: ~13 ms (PCIe Gen4 ×16)
- MPI AllReduce: ~a few ms (Ethernet, same node)
- H→D: ~13 ms

Total: ~26 ms per checkpoint call.  For checkpoints every 1000 events at
~100 evt/s, this is called every ~10 seconds — negligible.

For more frequent checkpoints or larger arrays, NCCL AllReduce (GPU-direct)
would be preferable.  `create_gpu_communicators()` already reserves
`bd_comm` "for XPCS AllReduce via NCCL" but NCCL is not yet wired in.

### 4. Shape and dtype constraints on cpu_result

`cpu_result` can be a Python scalar, NumPy scalar, or NumPy array.
**All BD ranks must pass cpu_result with the same shape and dtype.**

If different BD ranks have differently-shaped CPU results (e.g., variable-length
waveforms), a fixed-size buffer must be agreed on at `DataSource` construction.
For scalar diagnostics (GMD energy, IPM position) this is never an issue.

### 5. Partial result accumulator memory

After `join()`, the partial accumulator should typically be reset.
For GPU accumulators (CuPy arrays), this is a `arr.fill(0)` call.
The `join()` function itself does not reset accumulators — the user
controls this.

If `free_calib_bufs()` is called at the checkpoint to reclaim GPU memory,
any CuPy arrays derived from `calib_gpu` (views into the pre-allocated
slot buffer) would become invalid.  The user must ensure their accumulator
arrays are independent copies, not views.

### 6. Checkpoint alignment with batch boundaries

The `EventPool(n=4)` pipelines 4 batches ahead.  When the user calls
`join()` after event 1000, up to 4 × batch_size events may still be
in-flight in the pipeline.  The `join()` would include results from
events yielded up to event 1000, but the GPU kernels for events
1001–1320 (next 4 batches) might be running concurrently.

This is only an issue if the user holds references to in-flight
`GpuEventContext` objects.  For the pattern `if i_evt % 1000 == 0: join(accumulator, ...)` where the accumulator is updated each event, this is safe — the accumulator is updated after each `yield` returns, when the event has been fully processed by user code.

### 7. Non-GPU BD ranks

If the cluster runs with both GPU BD ranks and CPU BD ranks (mixed
topology), `join()` must handle heterogeneous result locations.  For now,
assume all BD ranks in the `bd_workers_comm` are GPU BD ranks.

---

## Implementation plan

| Step | Work | Complexity |
|---|---|---|
| 1. Create and expose `bd_workers_comm` | Build a BD-workers-only sub-communicator (excludes EB) in `_gpu_events_mpi()` using `bd_comm.Create_group()`; expose as `run.bd_workers_comm` or `ctx._bd_workers_comm` | Low |
| 2. `ctx.join(partial, op='sum')` method | Add to `GpuEventContext`; implements MPI AllReduce via D→H→MPI→H→D | Low-Medium |
| 3. `run.join(partial, op='sum')` alternative | Add to `RunParallel` for consistency with CPU path | Low |
| 4. NCCL AllReduce backend | Wire `nccl_comm` into `create_gpu_communicators()`; `ctx.join(..., backend='nccl')` | High |
| 5. Streaming join (non-blocking) | Submit AllReduce non-blocking; collect at next checkpoint | Medium |

Step 2 is the minimum viable implementation.  Steps 4–5 are optimisations.

---

## Open questions

1. **Does "every 1000 events" count locally** (each rank's own 1000th event,
   simplest) **or globally** (total across all ranks reaches 1000, requires
   a pre-AllReduce to check the total)?
   *Local count is recommended — matches the natural `i_evt % 1000 == 0` pattern.*

2. **What reduction operation?** Sum is the default and covers the majority
   of use cases (image sums, energy sums, hit counts).  Should `op` be a
   string enum (`'sum'`, `'max'`, `'mean'`) or an arbitrary callable?

3. **Should `join()` be synchronous** (blocks until all BD ranks reach the
   same checkpoint) **or asynchronous** (submits non-blocking AllReduce,
   user calls `.result()` later to collect)?
   *Synchronous is simpler to implement and reason about; async is better
   for overlapping computation with communication.*

4. **NCCL for GPU AllReduce?**  The `create_gpu_communicators()` function
   already reserves `bd_comm` "for XPCS AllReduce via NCCL" but NCCL is
   not yet wired in.  For the MFX Jungfrau 64 MB array, the MPI path costs
   ~26 ms (D→H→MPI→H→D).  Is this acceptable, or should NCCL be prioritised?

5. **How is `bd_comm` exposed to the user?**  Options:
   - `ctx.join(cpu_result, gpu_result)` — context carries the communicator
   - `run.join(cpu_result, gpu_result)` — run carries the communicator
   - Explicit `join(cpu, gpu, bd_comm=run.bd_workers_comm)` — user passes it

6. **What happens to in-flight EventPool batches** when `join()` is called?
   Up to 4 × batch_size events may be pipelined ahead.  The accumulator
   reflects only events yielded so far, not in-flight events.  Is this
   the intended semantics, or should `join()` drain the pipeline first?
