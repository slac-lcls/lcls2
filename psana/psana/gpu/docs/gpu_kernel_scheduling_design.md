# GPU Kernel Scheduling — Design Document

**Status:** Proposed — for team review
**Scope:** `psana.gpu` — scheduled kernel execution path
**Related:** `gpu_kernel_registry_design.md` (registry foundation)

---

## 1. Motivation

The current `GpuKernelRegistry` (see `gpu_kernel_registry_design.md`) is a
named function store: users register kernels, then manually call `launch_1d()`
or `run()` inside the event loop.  This requires the user to manage streams,
allocate output buffers, and handle slot-lease ordering by hand.

The **scheduled execution path** removes that burden by mirroring how psana
already handles calibration: the user declares *what* to run and *on what
data*; psana dispatches the kernels as part of its internal batch pipeline,
manages per-slot output buffers, and makes results available through the same
`ctx.get()` / `GPUResult` API that calibrated data uses.

### Comparison

| Concern | Calib today | Manual registry | **Scheduled path (proposed)** |
|---|---|---|---|
| Kernel expressed as | internal `fused_calib_gpu` | user CUDA source / callable | user CUDA source / callable |
| User declares intent at | `DataSource(gpu_det=...)` | `register_cuda(...)` | `register_cuda(...)` + `schedule_for(...)` |
| Dispatch | psana owns it | user calls `launch_1d()` in loop | **psana owns it** |
| Output buffer | per-slot VRAM, slot-leased | user allocates with `cp.empty()` | **per-slot VRAM, slot-leased** |
| Result access | `ctx.get('det.calib')` | raw `cp.ndarray` variable | **`ctx.get('output_key')`** |
| Multiple events in-flight | two-slot double-buffering | user must handle manually | **same two-slot pipeline** |

---

## 2. New API Surface

Three additions to `GpuKernelRegistry`:

```
GpuKernelRegistry
├── (existing) register_cuda(name, source, entry_point, ...)
├── (existing) register_callable(name, ...)
│
├── (NEW) schedule_for(name, *, input_key, output_key, output_dtype,
│                      output_shape, args_fn)   → KernelBinding
│
├── (NEW) get_bindings()                         → List[KernelBinding]
└── (NEW) get_bindings_for(input_key)            → List[KernelBinding]

KernelBinding  (NEW dataclass)
├── kernel_name:    str          name in the registry
├── input_key:      str          ctx.get() key that provides the input array
├── output_key:     str          ctx.get() key written by this kernel
├── output_dtype:   dtype        output element type
├── output_shape_fn: callable    input_shape → output_shape  (None = same)
└── args_fn:        callable     (inp_gpu, out_gpu, stream) → args tuple
                                 (cuda kernels only; None = default convention)
```

A new `GpuKernelExecutor` class (internal to psana) wraps a `KernelBinding`
and manages the per-slot VRAM output buffers for it.

---

## 3. Usage Examples

### 3.1 Single kernel — threshold filter

The simplest case: one custom kernel that zeroes pixels below a threshold,
producing a same-shape result available as `ctx.get('threshold')`.

```python
# ── Module level ─────────────────────────────────────────────────────────────
from psana.gpu import gpu_kernel_registry
import numpy as np

THRESHOLD_SRC = r"""
extern "C" __global__
void threshold(const float* __restrict__ in,
               float*       __restrict__ out,
               float thresh, unsigned long long n)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = (in[i] > thresh) ? in[i] : 0.0f;
}
"""

gpu_kernel_registry.register_cuda(
    'threshold',
    source      = THRESHOLD_SRC,
    entry_point = 'threshold',
)

gpu_kernel_registry.schedule_for(
    'threshold',
    input_key    = 'jungfrau.calib',
    output_key   = 'threshold',     # accessed as ctx.get('threshold')
    output_dtype = np.float32,
    output_shape = 'same',          # same shape as jungfrau.calib
    args_fn      = lambda inp, out, stream: (
                       inp.ravel(), out.ravel(),
                       np.float32(5.0),       # threshold value
                       np.uint64(inp.size),   # n
                   ),
)

# ── DataSource (unchanged) ───────────────────────────────────────────────────
from psana import DataSource
ds  = DataSource(exp='mfxl1016122', run=77, gpu_det='jungfrau')
run = next(ds.runs())

# ── Event loop ───────────────────────────────────────────────────────────────
for ctx in run.events():
    result = ctx.get('threshold')    # GPUResult — same API as jungfrau.calib
    arr    = result.on_cpu           # np.ndarray, blocks until D→H complete
    print(arr.shape, arr.max())
```

The user never calls `launch_1d()` or allocates an output buffer.

---

### 3.2 Two independent kernels on the same input

Both `threshold` and `panel_mean` operate on `jungfrau.calib`; they run
sequentially on the slot's CUDA stream after calibration completes.

```python
# ── Module level ─────────────────────────────────────────────────────────────
gpu_kernel_registry.register_cuda(
    'threshold', source=THRESHOLD_SRC, entry_point='threshold'
)
gpu_kernel_registry.schedule_for(
    'threshold',
    input_key    = 'jungfrau.calib',
    output_key   = 'threshold',
    output_dtype = np.float32,
    output_shape = 'same',
    args_fn      = lambda inp, out, s: (inp.ravel(), out.ravel(),
                                        np.float32(5.0), np.uint64(inp.size)),
)

@gpu_kernel_registry.register_callable('panel_mean')
def panel_mean(calib_gpu, out=None, stream=None):
    """Mean intensity per panel — reduces (n_segs, nrows, ncols) → (n_segs,)."""
    import cupy as cp
    with (stream or cp.cuda.Stream.null):
        return calib_gpu.mean(axis=(-2, -1))   # shape (n_segs,)

gpu_kernel_registry.schedule_for(
    'panel_mean',
    input_key    = 'jungfrau.calib',
    output_key   = 'panel_mean',
    output_dtype = np.float32,
    output_shape = lambda s: (s[0],),   # (n_segs,) from (n_segs, nrows, ncols)
)

# ── Event loop ───────────────────────────────────────────────────────────────
for ctx in run.events():
    thresh_result = ctx.get('threshold')
    means_result  = ctx.get('panel_mean')

    thresh_arr = thresh_result.on_cpu   # np.float32 (n_segs, nrows, ncols)
    means_arr  = means_result.on_cpu    # np.float32 (n_segs,)

    hit = means_arr.max() > 10.0
```

---

### 3.3 Chained kernels — output of one feeds the next

`peak_finder` takes the thresholded data (not raw calib) as its input.
The `input_key` for the second kernel is the `output_key` of the first.

```python
# ── Module level ─────────────────────────────────────────────────────────────
PEAK_FINDER_SRC = r"""
extern "C" __global__
void peak_finder(const float* __restrict__ in,
                 float*       __restrict__ out,
                 unsigned long long n)
{
    // simplified: mark pixels that are local maxima above background
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 || i >= n - 1) return;
    out[i] = (in[i] > in[i-1] && in[i] > in[i+1] && in[i] > 2.0f)
             ? in[i] : 0.0f;
}
"""

gpu_kernel_registry.register_cuda('threshold', source=THRESHOLD_SRC, entry_point='threshold')
gpu_kernel_registry.schedule_for(
    'threshold',
    input_key    = 'jungfrau.calib',
    output_key   = 'threshold',
    output_dtype = np.float32,
    output_shape = 'same',
    args_fn      = lambda inp, out, s: (inp.ravel(), out.ravel(),
                                        np.float32(5.0), np.uint64(inp.size)),
)

gpu_kernel_registry.register_cuda('peak_finder', source=PEAK_FINDER_SRC, entry_point='peak_finder')
gpu_kernel_registry.schedule_for(
    'peak_finder',
    input_key    = 'threshold',     # ← takes output of the threshold kernel
    output_key   = 'peaks',
    output_dtype = np.float32,
    output_shape = 'same',
    args_fn      = lambda inp, out, s: (inp.ravel(), out.ravel(), np.uint64(inp.size)),
)

# ── Event loop ───────────────────────────────────────────────────────────────
for ctx in run.events():
    peaks = ctx.get('peaks').on_cpu
    n_peaks = int((peaks > 0).sum())
```

psana resolves the dependency graph at `GpuEvents` setup time and
validates that `threshold` is scheduled before `peak_finder`.

---

### 3.4 Multiple detectors — same kernel, different inputs

The same registered kernel can be scheduled multiple times, once per detector,
by giving each binding a distinct `output_key`.

```python
# register once
gpu_kernel_registry.register_cuda('threshold', source=THRESHOLD_SRC, entry_point='threshold')

# schedule against two detectors
gpu_kernel_registry.schedule_for(
    'threshold',
    input_key    = 'jungfrau.calib',
    output_key   = 'jungfrau_threshold',
    output_dtype = np.float32,
    output_shape = 'same',
    args_fn      = lambda inp, out, s: (inp.ravel(), out.ravel(),
                                        np.float32(5.0), np.uint64(inp.size)),
)
gpu_kernel_registry.schedule_for(
    'threshold',
    input_key    = 'epix10k2m.calib',
    output_key   = 'epix_threshold',
    output_dtype = np.float32,
    output_shape = 'same',
    args_fn      = lambda inp, out, s: (inp.ravel(), out.ravel(),
                                        np.float32(3.0), np.uint64(inp.size)),
)

# DataSource with two GPU detectors
ds = DataSource(exp='mfxl1016122', run=77, gpu_det=['jungfrau', 'epix10k2m'])

for ctx in run.events():
    jf_thresh  = ctx.get('jungfrau_threshold').on_cpu
    ep_thresh  = ctx.get('epix_threshold').on_cpu
```

---

### 3.5 Python callable with reduction output

`roi_sum` reduces the calibrated image to a scalar per panel using CuPy.
psana copies the return value into the slot buffer so it is slot-lease
protected like all other results.

```python
ROI = (slice(None), slice(10, 20), slice(10, 20))  # all segs, 10×10 patch

@gpu_kernel_registry.register_callable('roi_sum')
def roi_sum(calib_gpu, stream=None):
    import cupy as cp
    with (stream or cp.cuda.Stream.null):
        return calib_gpu[ROI].sum(axis=(-2, -1))   # (n_segs,)

gpu_kernel_registry.schedule_for(
    'roi_sum',
    input_key    = 'jungfrau.calib',
    output_key   = 'roi_sum',
    output_dtype = np.float32,
    output_shape = lambda s: (s[0],),   # (n_segs,)
)

for ctx in run.events():
    roi_vals = ctx.get('roi_sum').on_cpu   # np.float32 (n_segs,)
```

---

### 3.6 Accessing results on GPU (zero-copy)

All scheduled kernel outputs support the same three accessors as `jungfrau.calib`:

```python
for ctx in run.events():
    result = ctx.get('threshold')

    # ── Option A: independent GPU copy (can outlive the event) ───────────────
    arr_gpu = result.on_gpu           # cp.ndarray, D→D copy; slot freed immediately

    # ── Option B: zero-copy view inside a stream-ordered context ─────────────
    stream = cp.cuda.Stream(non_blocking=True)
    with result.on_gpu_view(stream) as arr_view:
        # arr_view is the raw slot buffer — no copy
        # any work launched here on `stream` completes before the slot recycles
        reduced = arr_view.mean()
    # slot is safe to recycle after `stream` reaches this point

    # ── Option C: synchronous host copy ──────────────────────────────────────
    arr_cpu = result.on_cpu           # np.ndarray, blocks for D→H
```

`on_gpu_view` is preferred for in-loop GPU work: it avoids the D→D copy and
lets the user's downstream kernel share the same stream order as psana's
calibration pipeline without any explicit synchronization.

---

## 4. Execution Model

### 4.1 Kernel scheduling within a batch

psana processes events in *batches* (groups of N events packed by EventBuilder
into a GPUBAT1 wire-format message).  The batch is further split into
*subbatches* by `GpuEvents._split_subbatches()` based on available VRAM.

For each subbatch `EventPool.submit()` queues all GPU work on one non-blocking
CUDA stream, then records a single `result_ready` CUDA event that gates
**all** results — calibration and scheduled kernels alike.

The execution order on the stream for a subbatch with M events and two
scheduled kernels A and B (B depends on A):

```
CUDA stream (slot k)
─────────────────────────────────────────────────────────────────────────►

 calib       ┌──────────────────────────────────────────────────────┐
 kernel      │  event 0  │  event 1  │  …  │  event M-1            │
             └──────────────────────────────────────────────────────┘

 kernel A    ┌──────────────────────────────────────────────────────┐
 (threshold) │  event 0  │  event 1  │  …  │  event M-1            │
             └──────────────────────────────────────────────────────┘

 kernel B    ┌──────────────────────────────────────────────────────┐
 (peak_fdr)  │  event 0  │  event 1  │  …  │  event M-1            │
             └──────────────────────────────────────────────────────┘

             ▲
             result_ready.record()  ← single event, gates everything
```

Key properties:
- All kernels (calib, A, B) run on the *same slot stream*.  No cross-stream
  synchronization is needed.
- Calib for all M events is queued first, then kernel A for all M events, then
  kernel B.  This is safe because each event has a unique, non-overlapping
  slice of the slot buffer (see §4.3), so `calib[event 1]` cannot overwrite
  `calib[event 0]`'s output.
- Two subbatches run on different slot streams (slot 0 and slot 1 in the
  two-slot double-buffer) and overlap each other with the CPU deserialization
  path.

### 4.2 `EventPool.submit()` control flow

```
EventPool.submit(gv, gpu_read, cpu_evts, gpu_detectors, kernel_executors)
│
├── [1] Calibration loop  (existing)
│       for det_name, det_info in gpu_detectors:
│           for ec in gpu_det_obj.process_batch(gv, gpu_read, stream, slot):
│               gpu_results_by_ts[ec.timestamp]['det.calib'] = ec.calib_gpu
│               gpu_results_by_ts[ec.timestamp]['det.raw']   = ec.raw_gpu   # if present
│
├── [2] finalize_results hook  (existing — DetectorRouter image assembly)
│       gpu_results_by_ts = finalize_results(gpu_results_by_ts, cpu_evts, stream)
│
├── [3] Scheduled kernel loop  (NEW)
│       for output_key, executor in kernel_executors.items():
│           binding = executor.binding
│           for ts, ts_dict in gpu_results_by_ts.items():
│               inp = ts_dict.get(binding.input_key)  # may be calib OR another kernel's output
│               if inp is None:
│                   continue
│               out = executor.process_event(inp, stream, slot)
│               ts_dict[output_key] = out
│
├── [4] result_ready.record(stream)   ← AFTER all work (calib + scheduled kernels)
│
└── [5] Create SlotLease per (ts, key)
        all results in ts_dict get the same result_ready event
        each gets its own SlotLease for independent consumer tracking
```

`kernel_executors` is an `OrderedDict` built by `GpuEvents._setup_kernel_executors()`
in dependency order (§4.4), so a binding whose `input_key` is another kernel's
`output_key` always runs after that producer.

### 4.3 Per-slot buffer management

`GpuKernelExecutor` manages VRAM output buffers exactly as `GPUDetector`
manages its `_calib_slot_bufs`:

```
GpuKernelExecutor._output_slot_bufs
    slot 0 → cp.ndarray shape (total_segs_batch, nrows, ncols) float32
    slot 1 → cp.ndarray shape (total_segs_batch, nrows, ncols) float32

Each batch's M events get unique non-overlapping slices:
    event 0 → slot_buf[0 : seg_count_0]
    event 1 → slot_buf[seg_count_0 : seg_count_0 + seg_count_1]
    …
```

Buffers are **lazily allocated** on the first event of a run (shape not known
at registration time) and **grown in place** if a larger batch arrives.

Because all scheduled kernel outputs share the slot's `result_ready` event and
each has its own `SlotLease`, the slot is not recycled until *every* consumer —
D→H pipeline, `on_gpu_view` context exit, or explicit `on_gpu` copy — has
signalled completion.

### 4.4 Dependency resolution at setup time

`GpuEvents._setup_kernel_executors()` runs once per run (called from
`_setup_detectors()`).  It:

1. Reads all bindings from `gpu_kernel_registry.get_bindings()`.
2. Builds a directed acyclic graph: an edge `A → B` exists when
   `binding_B.input_key == binding_A.output_key`.
3. Topologically sorts the graph.  Raises `ValueError` on a cycle.
4. Validates that every `input_key` either refers to a detector result known
   to this run (e.g. `jungfrau.calib`) or to another binding's `output_key`.
5. Returns an `OrderedDict[output_key → GpuKernelExecutor]` in execution order.

Any binding whose `input_key` is not present for a given event (e.g. because
that detector had no data) is silently skipped for that event — matching the
behaviour of calibration itself.

---

## 5. Result Access Reference

### 5.1 `ctx.get(key)` — GPUResult

All scheduled-kernel results are accessed identically to calib results:

```python
result = ctx.get('threshold')   # GPUResult
```

`GpuEventContext.get()` does a dictionary lookup on `gpu_results_by_ts[ts]`;
no code change is needed there because scheduled kernel outputs are written
into the same dict as calib outputs.

### 5.2 GPUResult accessors

| Accessor | Copy | Stream sync needed | Recommended when |
|---|---|---|---|
| `.on_gpu` | D→D copy, independent array | No | Need array past this event iteration, or feed to multiple streams |
| `.on_gpu_view(stream)` | zero-copy slot buffer view | No (stream-ordered via CUDA event) | Run downstream GPU work on the same stream; most efficient |
| `.on_cpu` | blocking D→H (or async pinned if D2H pipeline active) | Yes (implicit) | Need NumPy array in Python |

### 5.3 Checking whether a result exists

A scheduled kernel only writes a result when its `input_key` was present for
the event.  Use `ctx.has('threshold')` to check before `ctx.get()`:

```python
for ctx in run.events():
    if ctx.has('threshold'):
        arr = ctx.get('threshold').on_cpu
```

### 5.4 Introspecting scheduled bindings

```python
from psana.gpu import gpu_kernel_registry

bindings = gpu_kernel_registry.get_bindings()
for b in bindings:
    print(f"{b.kernel_name}: {b.input_key} → {b.output_key}")

# threshold: jungfrau.calib → threshold
# peak_finder: threshold → peaks
# panel_mean: jungfrau.calib → panel_mean
```

---

## 6. Internal Class Summary

| Class / Method | File | Role |
|---|---|---|
| `KernelBinding` | `gpu_kernel_registry.py` | Dataclass: kernel name, input key, output key, dtype, shape fn, args fn |
| `GpuKernelRegistry.schedule_for()` | `gpu_kernel_registry.py` | Register a `KernelBinding`; append to `_bindings` list |
| `GpuKernelRegistry.get_bindings()` | `gpu_kernel_registry.py` | Return all bindings (all input keys) |
| `GpuKernelRegistry.get_bindings_for(key)` | `gpu_kernel_registry.py` | Return bindings whose `input_key == key` |
| `GpuKernelExecutor` | `gpu_kernel_registry.py` | Per-binding executor; holds `_output_slot_bufs[n_slots]`; calls `process_event()` |
| `GpuEvents._setup_kernel_executors()` | `gpu_events.py` | DAG sort; validate input keys; build `OrderedDict[key → executor]` |
| `EventPool.submit(..., kernel_executors)` | `gpu_stream.py` | New loop (step 3 above); writes outputs into `gpu_results_by_ts` |
| `GpuEventContext.get()` | `context.py` | Unchanged — dict lookup works for both calib and scheduled-kernel outputs |
| `GpuEventContext.has()` | `context.py` | New helper: `return key in self._gpu_results` |

---

## 7. MPI Considerations

### 7.1 Process-wide singleton and MPI ranks

Each MPI BD rank is a separate OS process.  `gpu_kernel_registry` is a
process-wide singleton, so each rank independently holds a copy of the
registered kernels and bindings.  There is no cross-rank state.

The user's analysis script (or an imported module) must call `register_cuda()`
and `schedule_for()` before `DataSource(...)` on every rank that will process
events — this is no different from registering CUDA kernels today.

### 7.2 Compilation cost

NVRTC compilation is per-process.  With N BD ranks on one node, each rank
compiles independently.  For large kernels, call `compile_all()` before the
event loop to pay the cost once at startup rather than on the first event:

```python
gpu_kernel_registry.compile_all()   # after all register_cuda() calls
ds = DataSource(...)
```

### 7.3 Leader / follower BD ranks sharing a GPU

When two BD ranks share one physical GPU (CUDA IPC mode), the calibration
constants are shared through `share_calib_between_gpu_peers()`.  Scheduled
kernel **output buffers are not shared** — each rank maintains its own
`GpuKernelExecutor._output_slot_bufs`.  This is correct: each rank processes
a different subset of events and needs independent output buffers.

### 7.4 VRAM budget

`GpuKernelExecutor` registers its output buffer allocations with `_GpuBudget`
under the category `user_kernel_slots`, following the same pattern as
`_calib_slot_bufs`.  The budget is checked in `_setup_kernel_executors()` to
surface out-of-memory conditions at setup time rather than mid-run.

---

## 8. Design Decisions

### Why `schedule_for()` separate from `register_cuda()`?

A single kernel can be scheduled multiple times (once per detector, or with
different thresholds) without re-registering the source.  Keeping registration
(source + compilation) separate from scheduling (data binding) avoids
duplicating CUDA source strings and compiled kernels.

### Why `args_fn` instead of fixed argument conventions?

CUDA kernel argument order is set by the user's own C++ signature.  A fixed
convention (e.g. always `(in, out, n)`) would break any kernel with additional
scalar parameters.  `args_fn` is a closure that captures constants naturally:

```python
args_fn = lambda inp, out, s: (inp.ravel(), out.ravel(),
                                np.float32(MY_THRESHOLD),
                                np.uint64(inp.size))
```

For callable kernels `args_fn` is not needed: the callable already has full
Python expressiveness to compute whatever it needs.

### Why run all calib events before all kernel-A events?

Kernel A's output for event 0 is in a unique, non-overlapping slice of the
slot buffer.  Running calib for event 1 cannot overwrite it.  Processing each
kernel over the entire batch before starting the next kernel keeps the code
simple, avoids per-event kernel-launch overhead from Python, and lets the GPU
driver pipeline more effectively.

### Why `result_ready` is recorded after all scheduled kernels?

All results — calib and user kernels — are on the same stream.  A single
`result_ready` event covers everything: consumers of `jungfrau.calib` and
consumers of `threshold` both wait on the same fence.  This is strictly
correct and adds zero extra synchronization.

### Why not a separate stream per scheduled kernel?

Using the slot's existing stream for all work (calib + scheduled kernels)
preserves the ordering guarantee without cross-stream events.  The GPU's
hardware scheduler overlaps independent kernels even on a single stream when
the occupancy headroom exists.

### `on_gpu_view` for scheduled-kernel outputs

Users can pass the zero-copy view of a scheduled kernel's output into a
*further* GPU kernel that they call manually, and the slot will not recycle
until the user's downstream kernel records its `_consumer_done` event via
`__exit__`.  This is identical to how `on_gpu_view` works for calib outputs
today.

---

## 9. Open Questions

1. **Scalar / aggregate outputs** — for kernels that reduce an entire image to
   a single float (total intensity, hit flag), the output buffer is
   `(n_slots, 1)` floats.  Should `schedule_for()` accept `output_shape='scalar'`
   as a convenience shorthand?

2. **Runtime args** — `args_fn` captures constants at schedule time.  If a
   user needs to change a threshold between runs (e.g. from a BeginRun
   configuration object), there is currently no path to update it.  Should
   `KernelBinding` support a `dynamic_args_fn(ctx) → tuple` variant that
   receives the `GpuEventContext` and can read per-run configuration?

3. **Eager binding validation** — `schedule_for()` currently records the
   binding but cannot validate `input_key` until `GpuEvents._setup_detectors()`
   runs (when the actual detector list is known).  Should `schedule_for()`
   accept an optional `run` argument to validate immediately?

4. **Kernel output as D2H pipeline input** — the automatic D→H pipeline
   (`_D2hPipeline`) currently only targets calib outputs.  Extending it to
   scheduled-kernel outputs so `result.on_cpu` is backed by pinned async
   transfers (rather than blocking `arr.get()`) would be a natural follow-on.

5. **`ctx.has()` vs `ctx.get()` returning `None`** — which convention is
   preferable for missing results when a kernel's input was absent for an event?
