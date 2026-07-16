# Lazy-Sync D→H Design

## Four design decisions

Three decisions together make the design correct, safe, and performant:

1. `on_gpu` always returns a D→D copy — safe by default, no lease needed
2. `on_gpu_view` is the explicit fast path — zero-copy, requires `release_after()`
3. Pool depth defaults to 2 — sufficient overlap for basic calib kernel
4. `_GpuBudget` — simple committed-bytes counter prevents OOM before `cp.empty()`

---

## The problem

GPU calibration results live in device VRAM.  To read them on the CPU you
need a Device→Host (D→H) PCIe transfer (~13 ms per Jungfrau event).

The naive approach blocks the event generator until the transfer is done,
which stalls the GPU pipeline:

```
generator                           user loop
─────────                           ─────────
retire batch
_yield_ready():
  issue D→H ... wait ... wait ...   ← GPU idle during transfer
  yield ctx  ──────────────────────► arr = ctx.get('det.calib').on_cpu
submit next batch
```

## The solution: issue early, sync late

Issue the D→H transfer the moment the calibration kernel completes,
yield the event context immediately (before the transfer finishes),
and let `on_cpu` do the wait only when the user actually asks for the data.

```
generator                           user loop
─────────                           ─────────
retire batch
_yield_ready():
  issue D→H ↓ async, returns ~0ms
  yield ctx  ──────────────────────► ... user does other work ...
submit next batch                  ► arr = ctx.get('det.calib').on_cpu
                                         │
                                         └─ wait for D→H here
GPU calibrates next batch ◄──────────────── (D→H overlaps with GPU work)
```

The key shift: the sync point moves from the **generator** to the **call
site**.  The GPU never idles waiting for PCIe.

---

## Full pipeline

```
DataSource(gpu_det='jungfrau', n_gpu_streams=2, gpu_d2h_chunk_size=10,
           gpu_memory_budget_gb=15)
                │
                ▼
        GpuEvents.__init__()
                │
                ├─── _GpuBudget(15 GB)  ◄──── budget.reserve() before every cp.empty()
                │         │                   GpuMemoryPressureError if over limit
                │         ▼
                │    KvikioGpuReader(n_slots=2, budget=budget)
                │         │  raw input slot buffers: uint8, grow lazily
                │         │
                │    GPUDetector(n_slots=2, budget=budget)
                │         │  calib_slot_bufs: float32, batch×segs×r×c
                │
                ├─── EventPool(n=2)
                │         │  2 non-blocking CUDA streams
                │         │  retire_next() syncs slot from 2 batches ago (instant)
                │
                ├─── SlotLease  ◄─── created per-event in EventPool.submit()
                │         │  calib_done CUDA event (after kernel)
                │         │  _d2h_done  CUDA event (after consumer)
                │         │  _needs_release flag (set by on_gpu_view)
                │
                └─── _D2hPipeline  ◄─── activated when gpu_d2h_chunk_size > 0
                          │  issues async D→H, yields events immediately
                          │  on_cpu syncs lazily at call site
```

---

## Three ways to get calibrated data

```python
# Choose based on use case:
arr = ctx.get('jungfrau.calib').on_gpu        # D→D copy  — safe, no ceremony
arr = ctx.get('jungfrau.calib').on_gpu_view   # zero-copy — fast, must release
arr = ctx.get('jungfrau.calib').on_cpu        # D→H       — transparent, numpy
```

### `on_gpu` — safe D→D copy

Returns an independent CuPy array.  The slot buffer can be recycled
immediately — no lease, no `release_after`, no user ceremony.

```python
for ctx in run.events():
    arr  = ctx.get('jungfrau.calib').on_gpu    # ~2 ms D→D copy
    hits = peak_finder(arr)
```

```
slot_buf (VRAM, slot 0)
    │
    └─► arr.copy()  ──────────────────────────────────────────────────────►
                      independent cp.ndarray in VRAM (not slot 0)
                      slot 0 recycled immediately — no lease registered
                      copy freed by Python GC after loop iteration
```

**SlotLease state after `on_gpu`:**
```
_needs_release = False
_d2h_done      = None
retire_next()  → recycle immediately ✓
```

### `on_gpu_view` — zero-copy view

Returns a view directly into the slot buffer.  Fastest path (~0 ms), but
**requires** calling `release_after(done_event)` after any downstream GPU
kernel.  If forgotten, `retire_next()` raises `RuntimeError` at slot-recycle
time — a loud failure, not silent corruption.

```python
stream = cp.cuda.Stream(non_blocking=True)

for ctx in run.events():
    result = ctx.get('jungfrau.calib')
    arr    = result.on_gpu_view                # zero-copy

    peak_finder(arr, stream=stream)

    done = cp.cuda.Event(disable_timing=True)
    stream.record(done)
    result.release_after(done)                 # slot safe after kernel
```

```
slot_buf (VRAM, slot 0)
    │
    └─► view (same object)  ──────────────────────────────────────────────►
                               user kernel reads directly from slot 0
                               MUST call release_after(done_event)

    retire_next(slot 0, 2 batches later):
        wait_until_safe_to_reuse():
            _d2h_done.synchronize()  ← blocks until kernel done ✓
        stream.synchronize()
        slot 0 recycled
```

**SlotLease state transitions:**
```
on_gpu_view called:   _needs_release = True,  _d2h_done = None
release_after(done):  _needs_release = True,  _d2h_done = done_event
retire_next():        done_event.synchronize() → recycle ✓

on_gpu_view, no release_after:
retire_next():        RuntimeError("release_after() was never called") ← loud failure
```

### `on_cpu` — transparent D→H

Returns a numpy array.  When `gpu_d2h_chunk_size > 0`, the D→H was already
issued by `_D2hPipeline` before the context was yielded — `on_cpu` waits
lazily only if the transfer hasn't completed yet.  `release_after` is called
automatically by the pipeline.

```python
ds = DataSource(..., gpu_d2h_chunk_size=10)

for ctx in run.events():
    arr = ctx.get('jungfrau.calib').on_cpu     # numpy, D→H transparent
    numpy_analysis(arr)
```

```
Time →   0ms       10ms      20ms      30ms      40ms      50ms
         │          │         │         │          │          │
GPU:     [── calib events 0-9 on stream_0 ──────────────────►]
                    │
             calib_done ← CUDA event recorded

_D2hPipeline._flush_chunk() at ~10ms:
    stream.wait_event(calib_done)
    memcpyAsync(pinned, slot_view) ─────────────────────────►[done ~50ms]
    done_event.record()
    for each event: result._pending_d2h = _PendingD2H(pslot, row, n_segs)
    yield ctx immediately ← D→H still in-flight

user calls on_cpu(ctx_0) at ~12ms:
    _pending_d2h.get()
        done_event.synchronize() ───────── blocks until ~50ms ────────────►
        pslot.arr[0].copy()    ← ~0.1ms
        return numpy array     ← at ~50ms

user calls on_cpu(ctx_1..9):
    done_event already fired → returns instantly
```

**SlotLease state:**
```
_D2hPipeline issues D→H:  registers done_event via lease.register_d2h_done()
retire_next():             done_event.synchronize() → recycle ✓
```

### Comparison

| Property | `on_gpu` | `on_gpu_view` | `on_cpu` |
|---|---|---|---|
| Returns | `cp.ndarray` copy | `cp.ndarray` view | `np.ndarray` |
| VRAM cost | +38 MB (copy) | 0 | 0 |
| Transfer | D→D ~2 ms | none | D→H ~13 ms or transparent |
| Slot recycled | immediately | after `release_after` fires | after D→H done |
| User action | none | `release_after()` required | none |
| If forgotten | n/a | `RuntimeError` at retire_next | n/a |

---

## How it works in the code

### 1. `_D2hPipeline._flush_chunk()` — issue and attach token

When `chunk_size` events have accumulated, the pipeline:

1. Issues `cudaMemcpyAsync` from the slot output view to a pinned host
   buffer (returns immediately, D→H runs on a separate CUDA stream).
2. Creates a `_PendingD2H` token for each event that carries:
   - a reference to the pinned slot (`_pslot`)
   - which row in the slot belongs to this event (`_row`)
   - the CUDA done-event (`_pslot.done_event`)
3. Attaches the token to `GPUResult._pending_d2h`.
4. **Yields the context immediately** — the transfer is still in-flight.

```
                  ┌──────────────────────────────────────┐
                  │  _D2hPipeline._flush_chunk()          │
                  │                                        │
  slot view ──►  memcpyAsync(pinned, view)  ← async      │
                  done_event.record()                      │
                  │                                        │
                  │  for each event i:                     │
                  │    result._pending_d2h =               │
                  │      _PendingD2H(pslot, row=i, ...)   │
                  │    yield ctx   ◄── immediately         │
                  └──────────────────────────────────────┘
```

### 2. `_PendingD2H.get()` — sync on demand

The token is consumed the first time `on_cpu` is called:

```python
class _PendingD2H:
    def get(self) -> np.ndarray:
        self._pslot.done_event.synchronize()   # wait for D→H
        data = self._pslot.arr[self._row, :self._n_segs].copy()
        self._pslot.dec_ref()                  # release pinned slot ref
        return data
```

The `_pslot` reference keeps the pinned buffer alive until every event
in the chunk has called `on_cpu`.  When the last event calls `dec_ref()`
the slot's reference count reaches zero and it is marked free for reuse.

### 3. `SlotLease.wait_until_safe_to_reuse()` — three outcomes

```python
def wait_until_safe_to_reuse(self):
    if self._d2h_done is not None:
        # on_cpu or on_gpu_view + release_after — wait for consumer
        self._d2h_done.synchronize()

    elif self._needs_release:
        # on_gpu_view called but release_after never called
        raise RuntimeError(
            "Slot cannot be recycled: on_gpu_view was accessed but "
            "release_after() was never called."
        )
    # else: on_gpu (copy) or no access — recycle immediately
```

### 4. `GPUResult.on_cpu` — three-path property

```python
@property
def on_cpu(self):
    # Path 1: already cached from a previous call — free.
    if self._pinned_cpu is not None:
        return self._pinned_cpu

    # Path 2: lazy sync — pipeline issued D→H before yielding.
    if self._pending_d2h is not None:
        self._pinned_cpu  = self._pending_d2h.get()   # waits here
        self._pending_d2h = None
        return self._pinned_cpu

    # Path 3: fallback — no pipeline active (gpu_d2h_chunk_size=0).
    if self._stream is not None:
        self._stream.synchronize()
    return self._arr.get()
```

---

## What happens if the user never calls on_gpu, on_gpu_view, or on_cpu

### Path 1 — no access at all

```python
for ctx in run.events():
    pass
```

`lease._d2h_done` stays `None`, `_needs_release` stays `False`.
`retire_next()` finds neither set and recycles the slot immediately.
**No problem.**

### Path 2 — `on_gpu` accessed, result discarded

```python
for ctx in run.events():
    arr = ctx.get('jungfrau.calib').on_gpu     # copy made
    # arr goes out of scope at next iteration
```

The copy is GC'd when `arr` goes out of scope.  The slot was never
locked — it was recycled immediately after the copy.  **No problem.**

### Path 3 — `on_gpu_view` accessed, `release_after` never called

```python
for ctx in run.events():
    arr = ctx.get('jungfrau.calib').on_gpu_view
    # forgot release_after
```

`lease._needs_release = True`, `lease._d2h_done = None`.
`retire_next()` at slot-recycle time raises `RuntimeError`.
**Loud failure — not silent corruption.**

### Path 4 — D→H pipeline, `on_cpu` never called

```python
ds = DataSource(..., gpu_d2h_chunk_size=10)
for ctx in run.events():
    pass   # _pending_d2h set but on_cpu never called
```

`_PendingD2H.__del__()` calls `pslot.dec_ref()` when the context is GC'd.
The slot's D→H completes and it is freed.  Correct but **wasted PCIe bandwidth**.

### Safe usage rules

| Pattern | Safe? |
|---|---|
| `on_gpu` in same iteration | Yes — copy is independent |
| `on_gpu_view` + `release_after` in same iteration | Yes |
| `on_gpu_view`, no `release_after` | RuntimeError at retire_next |
| `on_cpu` in same iteration | Yes |
| Ignore all results | Yes — GC handles cleanup |
| Collect all contexts, call `on_cpu` later | Unsafe if `n_events > max_inflight × chunk_size` |

---

## Batch-boundary flush

If `batch_size % chunk_size != 0` some events are left in `_chunk_buf`
at the end of a batch.  `_yield_ready()` calls `_flush_d2h_pipelines()`
after every batch so no event is stranded:

```
batch_size=15, chunk_size=10

  events 0-9  → chunk full → D→H issued → yield immediately
  events 10-14 → partial chunk (5 events)
  end of batch → _flush_d2h_pipelines()
                   └─ pipe.flush() → D→H issued for 5 events → yield
```

---

## Pool depth = 2

Pool_depth=2 fully hides the calibration kernel behind I/O.
Pool_depth=4 adds no parallelism (one NVMe read in-flight at a time)
and doubles the slot-buffer VRAM cost.

```
pd=2 (Jungfrau, bs=20):  2 × 760 MB = 1.5 GB   (sufficient)
pd=4:                     4 × 760 MB = 3.0 GB   (2× waste, was prior default)
```

---

## Memory budget — `_GpuBudget`

A simple committed-bytes counter prevents OOM from silently crashing the
MPI job.  Called before every `cp.empty()` in `GPUDetector` and
`KvikioGpuReader`.  No active-lease byte tracking is needed because
correctness is enforced by `wait_until_safe_to_reuse()`, not the budget.
The budget only prevents OOM:

```python
class _GpuBudget:
    def reserve(self, n):
        if self._committed + n > self._limit:
            cp.get_default_memory_pool().free_all_blocks()   # try pool flush
        if self._committed + n > self._limit:
            raise GpuMemoryPressureError(
                f"Need {n/1e9:.2f} GB, committed {committed/1e9:.2f} GB, "
                f"limit {limit/1e9:.2f} GB. Reduce batch_size or n_gpu_streams."
            )
        self._committed += n

    def release(self, n):
        self._committed = max(0, self._committed - n)
```

Default limit: `device_total_bytes / n_bd_ranks` (auto-detected at BeginRun).
Configurable: `DataSource(gpu_memory_budget_gb=15)`.

Both `GPUDetector` and `KvikioGpuReader` receive the **same** `_GpuBudget`
instance so that calib-slot and raw-input allocations are counted together
against a single per-BD limit.

---

## DataSource parameters

```python
DataSource(
    gpu_det              = 'jungfrau',
    n_gpu_streams        = 2,     # pool depth — default 2 (was 4)
    gpu_d2h_chunk_size   = 10,    # transparent on_cpu D→H — default 0 (disabled)
    gpu_memory_budget_gb = 15,    # per-BD VRAM limit — default auto
)
```

---

## Performance results (sdfampere, CPU-fallback I/O, 1000 events, 2 BD ranks, pd=2)

```
Configuration                        kHz        hot_ms
────────────────────────────────     ─────────  ───────
GPU baseline  bs= 1  pd=2            0.236 kHz  0.056 ms
GPU baseline  bs=10  pd=2            0.264 kHz  0.031 ms  ← ceiling (no D→H)
GPU baseline  bs=20  pd=2            0.276 kHz  0.028 ms
_D2hPipeline  chunk= 1  bs=20        0.114 kHz  14.6 ms   ← D→H cost per event
_D2hPipeline  chunk=10  bs=20        0.126 kHz  13.6 ms   ← lazy sync working

LCLS-II beam rate: 0.120 kHz  ← GPU baseline at bs≥10 exceeds beam rate
```

`chunk=1 ≈ chunk=10` hot latency confirms the lazy-sync design is correct:
the 13–15 ms is the D→H transfer time itself, not framework overhead.
I/O path is CPU-fallback (Lustre → CPU DRAM → GPU VRAM via cudaMemcpy);
true GDS (NVMe → GPU direct) would eliminate the CPU-DRAM hop.

---

## Files changed

| File | Change |
|---|---|
| `context.py` | `on_gpu` → D→D copy; `on_gpu_view` → view + `mark_needs_release`; `release_after()` → registers event on lease; `SlotLease._needs_release` + `RuntimeError` in `wait_until_safe_to_reuse()` |
| `gpu_budget.py` | New — `_GpuBudget` + `GpuMemoryPressureError` |
| `gpu_calib.py` | `GPUDetector.__init__` accepts `budget=`, `n_slots=2` default; `cp.empty()` guarded by `budget.reserve()` |
| `gpu_kvikio_read.py` | Same — `budget=`, `n_slots=2`, `cp.empty()` guarded |
| `gpu_events.py` | Creates `_GpuBudget.auto()` at init, passes shared instance to reader and detectors |
| `ds_base.py` | `n_gpu_streams` default 4→2; `gpu_memory_budget_gb` and `gpu_d2h_chunk_size` params added and forwarded to `DsParms` |

---

## Deferred

| Feature | When |
|---|---|
| Byte-bounded subbatches | Phase 3 — requires concurrent NVMe reads |
| Backpressure to EB | Phase 3 — follows subbatch admission |
| Shared-GPU coordination across BD ranks | Phase 4 |
| True GDS (NVMe → GPU direct) | Infrastructure — Lustre/GPFS doesn't support cuFile |

---

## Implementation phase status

### Phase 0 — Measurement and accounting
**Done.**
- `GPUDetector.memory_bytes()` — constants, geometry, calib_slots, raw_slots
- `KvikioGpuReader.memory_bytes()` — raw_input slots
- `_D2hPipeline.pinned_bytes()` — pinned host memory
- `_GpuMemStats` dataclass + `GpuEvents.log_memory()` — snapshot + high-water marks
- Called automatically at setup, first batch, and EndRun

### Phase 1 — Correct slot ownership
**Done.**
- `SlotLease` carries `calib_done` CUDA event and `_d2h_done` token
- `EventPool.submit()` records `calib_done` and creates one lease per event
- `EventPool.retire_next()` calls `wait_until_safe_to_reuse()` — generator
  advancement alone no longer recycles a slot
- `_PendingD2H.get()` calls `dec_ref()` to signal the slot is safe to reuse

### Phase 2 — Bounded asynchronous D→H
**Done.**

| Requirement | Status |
|---|---|
| Direct async D→H from slot view to pinned chunk | Done — `_D2hPipeline._flush_chunk()` |
| Separate logical join_size from physical chunk bytes | Done — `gpu_d2h_chunk_size` |
| Partial-tail flush at EndRun / BeginStep | Done |
| Partial-tail flush at batch boundary | Done |
| No full-size D2D join buffer | Done |
| Lazy sync — D→H overlaps with user processing | Done — `_pending_d2h` / `on_cpu` |
| `on_gpu` safe by default (D→D copy) | Done |
| `on_gpu_view` + `release_after` + `RuntimeError` safety | Done |
| `_GpuBudget` simple committed-bytes counter | Done |
| Pool depth default 2 | Done |
| `gpu_memory_budget_gb` enforcement | Done |

### Phase 3 — Byte-bounded subbatches
**Not done.**
- No memory estimation in `GPUDetector`
- No partitioning of one EB batch into byte-bounded GPU subbatches
- No backpressure queue — `GpuEvents` requests EB batches unconditionally
- `KvikioGpuReader` buffers grow without quota checks

### Phase 4 — Shared-GPU coordination
**Not done.**
- No aggregate budget enforcement across BD ranks sharing one A100
- No per-BD fairness or diagnostics

---

## Validation test coverage

| Test requirement | Status |
|---|---|
| Slot cannot be recycled while D→H in flight | Done — `test_retire_next_waits_for_d2h_before_recycle` |
| Generator advancement alone does not release a lease | Done — `test_generator_advancement_alone_does_not_release` |
| D→H completion token controls release | Done — `test_d2h_registered_calls_synchronize` |
| Multiple D→H chunks produce correct ordered join | Done — `test_on_cpu_returns_correct_data` |
| BeginStep and EndRun flush partial joins | Done — `test_pipeline_flush_partial` |
| `on_gpu` returns independent copy | Done — `test_on_gpu_returns_independent_copy` |
| `on_gpu_view` raises RuntimeError when release_after forgotten | Done — `test_retire_next_raises_if_release_after_forgotten` |
| Budget check prevents OOM before cp.empty() | Done — `TestGpuBudget` (5 tests) |
| Subbatch estimates stay within budget | Not done — Phase 3 |
| Variable event sizes split correctly | Not done — Phase 3 |
| Multiple BDs cannot exceed aggregate GPU budget | Not done — Phase 4 |
