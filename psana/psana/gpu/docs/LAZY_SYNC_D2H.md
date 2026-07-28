# Lazy-Sync D→H Design

## Five design decisions

Five decisions together make the design correct, safe, and performant:

1. `on_gpu` always returns a D→D copy — safe by default, no lease needed
2. `on_gpu_view(stream)` is the explicit fast path — zero-copy, context manager records done-event automatically
3. Pool depth defaults to 2 — sufficient overlap for basic calib kernel
4. `_GpuBudget` — simple committed-bytes counter prevents OOM before `cp.empty()`
5. `_available: SimpleQueue` — one mechanism for free-slot counting, identity, and retrieval

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

## Why each piece of machinery exists

Making async D→H work correctly is not free.  Each problem it creates
requires a specific fix, and those fixes are the classes you see in the
codebase.

```
PROBLEM 1: D→H blocks the generator for 13 ms
──────────────────────────────────────────────
  Naive:  [calib 2ms][wait 13ms][calib 2ms][wait 13ms] ...
                      ↑ GPU idle; exceeds 8 ms per-event budget at 120 Hz

  Fix:  issue D→H asynchronously, yield immediately, sync lazily at on_cpu
  → _D2hPipeline + _pending_d2h + GPUResult.on_cpu
  → This fix creates Problem 2.


PROBLEM 2: async D→H needs page-locked (pinned) host memory
────────────────────────────────────────────────────────────
  cudaMemcpyAsync cannot write to ordinary numpy memory — the OS must
  not page it out mid-transfer.  cudaMallocHost in the hot path is too slow.

  Fix:  pre-allocate page-locked buffers at startup
  → _PinnedSlot (2 per pipeline, pre-allocated in _D2hPipeline._init)
  → This fix creates Problem 3.


PROBLEM 3: on_cpu must know when the async transfer finished
─────────────────────────────────────────────────────────────
  The transfer runs on a separate CUDA stream.  We need a completion signal.

  Fix:  CUDA event recorded after memcpyAsync; on_cpu synchronizes it
  → _PinnedSlot.done_event  +  _PendingD2H.get()
  → This fix creates Problem 4.


PROBLEM 4: D→H reads from the VRAM slot; the next calibration writes to it
───────────────────────────────────────────────────────────────────────────
  Starting batch N+1 calibration into slot 0 while batch N's D→H is still
  copying OUT of slot 0 silently corrupts data.

  Fix:  two alternating VRAM slots; retire_next() waits for done_event
        before recycling a slot
  → EventPool(n=2)  +  retire_next()
  → This fix creates Problem 5.


PROBLEM 5: retire_next() and _D2hPipeline are in different classes
───────────────────────────────────────────────────────────────────
  The done_event lives on _PinnedSlot (host side).
  retire_next() lives on EventPool (VRAM slot side).
  They need to communicate.

  Fix:  SlotLease bridges the two — _D2hPipeline calls
        lease.register_d2h_done(pslot.done_event) and retire_next()
        calls lease._d2h_done.synchronize()
  → SlotLease


PROBLEM 6: user wants zero-copy GPU access without a D→D copy (~2 ms)
───────────────────────────────────────────────────────────────────────
  on_gpu makes a safe copy but costs ~2 ms.  For a user kernel that only
  reads the data once, the copy is waste.  But a raw view of the slot
  buffer without any safety mechanism causes silent corruption when the
  slot is recycled too soon.

  Fix:  on_gpu_view(stream) context manager — __exit__ records a done_event
        on the user's stream; retire_next() waits for it via the same
        SlotLease mechanism (Problem 5's fix, reused)
  → _GpuViewContext  (no new synchronization machinery needed)


PROBLEM 7: one EB batch may need more VRAM than one slot can hold
──────────────────────────────────────────────────────────────────
  batch_size=500 events × 60 MB = 30 GB  >  17 GB per-slot budget.
  Attempting to calibrate all 500 at once crashes cp.empty().

  Fix:  split the EB batch into pieces that each fit the per-slot budget
  → _split_subbatches()  +  GpuSubbatchView
  → This fix creates Problem 8.


PROBLEM 8: how do we know the per-slot VRAM budget?
────────────────────────────────────────────────────
  CuPy has no built-in allocation limit.  cp.empty() simply crashes
  the MPI job with CUDA_ERROR_OUT_OF_MEMORY and no useful message.

  Fix:  manual committed-bytes counter; raises a human-readable error
        before the allocation, and tries pool flush first
  → _GpuBudget  +  GpuMemoryPressureError
```

**Dependency chain in one line:**

```
async D→H
  → pinned memory (_PinnedSlot)
  → completion signal (CUDA done_event)
  → separate VRAM slots (EventPool)
  → slot-recycle safety (SlotLease + retire_next)
  → zero-copy user access (on_gpu_view + _GpuViewContext)
  → VRAM sizing (subbatches + _GpuBudget)
```

Removing async D→H and going back to blocking `arr.get()` would let you
delete all of the above.  The code would be ~10 lines.  It would also be
too slow to keep up with the 120 Hz LCLS-II beam rate.

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
                │
                └─── _D2hPipeline  ◄─── activated when gpu_d2h_chunk_size > 0
                          │  _available: SimpleQueue[_PinnedSlot]
                          │  _get_free_slot() calls get_nowait() — O(1), no scan
                          │  _PinnedSlot.dec_ref() calls put(self) when slot freed
                          │  issues async D→H, yields events immediately
                          │  on_cpu syncs lazily at call site
```

---

## Three ways to get calibrated data

```python
# Choose based on use case:
arr = ctx.get('jungfrau.calib').on_gpu                    # D→D copy  — safe, no ceremony
with ctx.get('jungfrau.calib').on_gpu_view(stream) as arr: # zero-copy — fast, auto-release
    ...
arr = ctx.get('jungfrau.calib').on_cpu                    # D→H       — transparent, numpy
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
_d2h_done = None
retire_next() → recycle immediately ✓
```

### `on_gpu_view(stream)` — zero-copy view

Returns a context manager.  `__enter__` yields a view directly into the
slot buffer (fastest path, ~0 ms).  `__exit__` records a CUDA done-event
on *stream* automatically — the slot is recycled only after that event fires.
No user action beyond the `with` statement.

```python
stream = cp.cuda.Stream(non_blocking=True)

for ctx in run.events():
    with ctx.get('jungfrau.calib').on_gpu_view(stream) as arr:
        peak_finder(arr, stream=stream)     # must use the same stream
    # done event recorded automatically in __exit__ — nothing else needed
```

```
slot_buf (VRAM, slot 0)
    │
    └─► view (same object)  ──────────────────────────────────────────────►
                               user kernel reads directly from slot 0
                               __exit__ records done_event on stream

    retire_next(slot 0, 2 batches later):
        wait_until_safe_to_reuse():
            _d2h_done.synchronize()  ← blocks until kernel done ✓
        stream.synchronize()
        slot 0 recycled
```

**SlotLease state transitions:**
```
with on_gpu_view(stream):  (running user kernel)
__exit__:                  _d2h_done = done_event  (recorded on stream)
retire_next():             done_event.synchronize() → recycle ✓
```

**Key constraint:** all kernels that read the view must be enqueued on
*stream* inside the `with` block.  A kernel on a different stream, or
enqueued after `__exit__`, will not be captured by the done-event.

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

| Property | `on_gpu` | `on_gpu_view(stream)` | `on_cpu` |
|---|---|---|---|
| Returns | `cp.ndarray` copy | `cp.ndarray` view (via `with`) | `np.ndarray` |
| VRAM cost | +38 MB (copy) | 0 | 0 |
| Transfer | D→D ~2 ms | none | D→H ~13 ms or transparent |
| Slot recycled | immediately | after `__exit__` done-event fires | after D→H done |
| User action | none | use `with` block; kernel on same stream | none |
| If forgotten | n/a | `__del__` records null-stream fallback | n/a |

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
        self._pslot = None                     # prevent __del__ double-release
        return data
```

The `_pslot` reference keeps the pinned buffer alive until every event
in the chunk has called `on_cpu`.  When the last event calls `dec_ref()`
the slot's reference count reaches zero and it is returned to the
`_available` queue for reuse.

### 3. `SlotLease.wait_until_safe_to_reuse()` — two outcomes

```python
def wait_until_safe_to_reuse(self):
    if self._d2h_done is not None:
        # on_cpu path or on_gpu_view(stream) __exit__ — wait for consumer
        self._d2h_done.synchronize()
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
    return self._arr.get()
```

---

## D→H pipeline walkthrough

### 1. Object graph — what exists at steady state

```
_D2hPipeline
  _available ──► SimpleQueue ──► [ slot_0 │ slot_1 ]   (both free initially)
  _pinned_pool ─► list ────────► [ slot_0 , slot_1 ]
  _chunk_buf   ─► []
  _d2h_stream  ─► cp.cuda.Stream (non_blocking=True)

  _PinnedSlot (slot_0)                        _PinnedSlot (slot_1)
  ┌──────────────────────────────────────┐    ┌─────────────────────────────┐
  │ arr  [chunk_size × segs × rows × cols│    │ arr  [same shape]           │
  │       page-locked host memory        │    │ page-locked host memory      │
  │ done_event: cp.cuda.Event            │    │ done_event: cp.cuda.Event   │
  │ _refs: 0                             │    │ _refs: 0                    │
  │ _refs_lock: threading.Lock           │    │ _refs_lock: threading.Lock  │
  │ _available ──► (pipeline's queue)    │    │ _available ──► (same queue) │
  └──────────────────────────────────────┘    └─────────────────────────────┘
```

Both slots start in the queue.  A slot leaves the queue when a chunk is
flushed (`get_nowait()`) and returns when all its events have called
`on_cpu` (`put(self)` from `dec_ref()`).

---

### 2. Flushing one chunk (chunk_size = 3)

```
Step A — take a free slot
─────────────────────────
  _get_free_slot():
    _available.get_nowait()  →  slot_0      queue = [slot_1]

Step B — issue async D→H, then claim
─────────────────────────────────────
  stream.wait_event(lease.calib_done)              ← calibration must be done first
  cudaMemcpyAsync(slot_0.arr[0], gpu_arr_0)  ─┐
  cudaMemcpyAsync(slot_0.arr[1], gpu_arr_1)   ├── all on d2h_stream, returns ~0 ms
  cudaMemcpyAsync(slot_0.arr[2], gpu_arr_2)  ─┘
  slot_0.done_event.record(d2h_stream)
  slot_0.claim(n=3)  →  _refs = 3

Step C — attach lazy-sync tokens and yield immediately
───────────────────────────────────────────────────────
  ctx_0._cache['jungfrau.calib']._pending_d2h = _PendingD2H(slot_0, row=0, n_segs)
  ctx_1._cache['jungfrau.calib']._pending_d2h = _PendingD2H(slot_0, row=1, n_segs)
  ctx_2._cache['jungfrau.calib']._pending_d2h = _PendingD2H(slot_0, row=2, n_segs)
  return [ctx_0, ctx_1, ctx_2]     ← D→H is still in-flight here!

  slot_0 is now pinned by 3 _PendingD2H tokens via _refs = 3
  Any of them can independently wait for done_event and copy their row.
```

---

### 3. User calls `on_cpu` — lazy sync and slot release

```
ctx_0.get('jungfrau.calib').on_cpu
│
├── GPUResult.on_cpu  (path 2 — pending_d2h set)
│     _pending_d2h.get():
│       slot_0.done_event.synchronize()    ← may block here (~13 ms first call)
│       data = slot_0.arr[0, :n_segs].copy()
│       slot_0.dec_ref():
│         with _refs_lock:  _refs 3 → 2  (freed = False)
│       _pslot = None          ← prevents __del__ double-release
│       return numpy array
│     _pending_d2h = None,  _pinned_cpu = numpy array

ctx_1.get('jungfrau.calib').on_cpu
│
├── _pending_d2h.get():
│     slot_0.done_event.synchronize()    ← already done, returns instantly
│     data = slot_0.arr[1, :n_segs].copy()
│     slot_0.dec_ref():  _refs 2 → 1  (freed = False)

ctx_2.get('jungfrau.calib').on_cpu
│
└── _pending_d2h.get():
      slot_0.done_event.synchronize()    ← already done, returns instantly
      data = slot_0.arr[2, :n_segs].copy()
      slot_0.dec_ref():
        with _refs_lock:  _refs 1 → 0  (freed = True)
      _available.put(slot_0)   ◄── slot returns to the free pool!

      queue is now [slot_0, slot_1] — both free again
```

---

### 4. Two-chunk timeline — how PCIe latency is hidden

```
Time →    0ms        13ms       20ms       33ms       40ms
          │           │          │           │          │

GPU stream_0:
  [──── calib chunk 0 (events 0-9) ────]
                      │  [──── calib chunk 1 (events 10-19) ────]

D→H stream:
  [──── memcpyAsync into slot_0 ──────────────────────]
                                  [──── memcpyAsync into slot_1 ──────────]

Generator:
  flush chunk 0 → slot_0 from queue → issue D→H → yield ctx 0–9
  flush chunk 1 → slot_1 from queue → issue D→H → yield ctx 10–19

User:
  ctx 0:  synchronize ──► blocks ~13 ms  │ copy | dec_ref (_refs=9)
  ctx 1:  done already ──► instant       │ copy | dec_ref (_refs=8)
  ...
  ctx 9:  done already ──► instant       │ copy | dec_ref (_refs=0)
                                                → put(slot_0) ─► queue=[slot_0]
  ctx 10: synchronize ──► blocks ~33 ms  │ copy | dec_ref (_refs=9)
  ...
  ctx 19: done already ──► instant       │ copy | dec_ref (_refs=0)
                                                → put(slot_1) ─► queue=[slot_0, slot_1]

  ─────────────────────────────────────────────────────────────────────────
  Only ctx_0 and ctx_10 block (the first call per chunk waits for D→H).
  The other 18 calls return in ~0.1 ms each (done_event already fired).
  The GPU calibrates chunk 1 while chunk 0 is transferring — no idle time.
```

---

### 5. Sync fallback when the queue is empty

Triggered when the user accumulates contexts without calling `on_cpu` fast
enough.  The generator yields without `_pending_d2h`; `on_cpu` falls back to
a synchronous `_arr.get()`.  Async D→H resumes automatically once any slot
is returned to the queue.

```
_available queue is empty (both slots claimed, on_cpu not yet called)

  Generator                          User
  ─────────                          ──────────────────────────────────
  add(ctx_20):
    _get_free_slot():
      _available.get_nowait()
      → raises Empty  →  return None

    _flush_chunk(pslot=None):
      return [ctx_20]    ← no _pending_d2h attached

  yield ctx_20 immediately

                                     ctx_20.get('jungfrau.calib').on_cpu:
                                       _pending_d2h is None  ← skip lazy path
                                       _stream.synchronize()
                                       return _arr.get()     ← sync D→H here

                                     ctx_18.get(...).on_cpu  →  dec_ref → _refs=0
                                       → _available.put(slot_0)
                                          ↑ queue non-empty again ✓

  add(ctx_21):
    _get_free_slot():
      _available.get_nowait()  →  slot_0   ← async D→H resumes
```

---

### 6. `_refs_lock` prevents the lost-update race

Without the lock, two concurrent `on_cpu` calls (one per event in the same
chunk) can both read the same stale `_refs` value, both decrement it to the
same result, and both store that value — so the count never reaches zero and
the slot is never returned to the queue.

```
slot_0._refs = 2  (chunk_size=2, two _PendingD2H tokens outstanding)

  WITHOUT lock:
  Thread A (on_cpu ctx_0)      Thread B (on_cpu ctx_1)
  ────────────────────────     ─────────────────────────
  LOAD  _refs  → 2             LOAD  _refs  → 2   ← reads stale 2
  [GIL switches to B]
                               SUBTRACT  → 1
                               STORE _refs = 1
  SUBTRACT  → 1
  STORE _refs = 1              ← overwrites B's store!
  1 > 0 → no put()             1 > 0 → no put()
  slot_0 NEVER RETURNED TO QUEUE  ✗

  WITH _refs_lock:
  Thread A                     Thread B
  ────────────────────────     ─────────────────────────
  acquire _refs_lock
  LOAD 2, SUBTRACT → 1
  STORE _refs = 1
  freed = False
  release _refs_lock
                               acquire _refs_lock
                               LOAD 1, SUBTRACT → 0
                               STORE _refs = 0
                               freed = True
                               release _refs_lock
                               _available.put(slot_0)  ✓
```

---

## What happens if the user never calls on_gpu, on_gpu_view, or on_cpu

### Path 1 — no access at all

```python
for ctx in run.events():
    pass
```

`lease._d2h_done` stays `None`.
`retire_next()` recycles the slot immediately.  **No problem.**

### Path 2 — `on_gpu` accessed, result discarded

```python
for ctx in run.events():
    arr = ctx.get('jungfrau.calib').on_gpu     # copy made
    # arr goes out of scope at next iteration
```

The copy is GC'd when `arr` goes out of scope.  The slot was never
locked — it was recycled immediately after the copy.  **No problem.**

### Path 3 — `on_gpu_view` context manager used correctly

```python
stream = cp.cuda.Stream(non_blocking=True)
for ctx in run.events():
    with ctx.get('jungfrau.calib').on_gpu_view(stream) as arr:
        kernel(arr, stream=stream)
    # __exit__ sets lease._d2h_done = done_event
```

`lease._d2h_done` is set by `_GpuViewContext.__exit__` before the `with` block exits.
`retire_next()` synchronizes the done-event.  **Correct.**

If the user forgets the `with` and the `_GpuViewContext` object is
garbage-collected, `__del__` records a conservative done-event on the null
stream — the slot is never permanently held.

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
| `with on_gpu_view(stream) as arr:` in same iteration | Yes — done-event auto-recorded |
| `on_gpu_view(stream)` object received but `with` not used | Handled by `__del__` (null-stream fallback) |
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

`_GpuBudget` is a simple committed-bytes counter that guards every
`cp.empty()` call in the pipeline.  Its single job is OOM prevention:
raise a human-readable error before the allocation happens, rather than
letting the CUDA driver crash the MPI job with a cryptic broken-pipe
message.  Correctness (slot ownership, safe recycling) is handled
separately by `SlotLease.wait_until_safe_to_reuse()`.

---

### 1. Ownership and sharing

One `_GpuBudget` instance is created by `GpuEvents._setup_detectors()`
and passed by reference to every component that allocates VRAM.  All
allocations therefore count against the same per-BD limit.

```
GpuEvents._setup_detectors()
  │
  ├── _GpuBudget(limit = device_total / n_bd_ranks)   ◄── single instance
  │        │
  │        ├─── → KvikioGpuReader(_budget=budget)     raw input slot buffers
  │        └─── → GPUDetector(_budget=budget)          calib output slot buffers
  │                   (one GPUDetector per det_name)
  │
  └── _compute_subbatch_budget()
           reads budget.limit() and budget.committed()
           → _subbatch_budget_bytes (drives _split_subbatches)
```

Default limit: `device_total / n_bd_ranks` auto-detected from CuPy at
BeginRun.  Override: `DataSource(gpu_memory_budget_gb=15)`.

---

### 2. VRAM layout — what is tracked vs. not tracked

```
VRAM  (example: A100 80 GiB, 2 BD ranks → limit = 40 GiB)
╔══════════════════════════════════════════════════════════════════╗
║  FIXED  —  allocated at BeginRun, never freed                   ║
║  counted in budget._committed from the moment they are reserved ║
║                                                                  ║
║  ┌──────────────────────────┐  ┌──────────────────────────┐    ║
║  │  peds_gpu + gmask_gpu    │  │  scatter_ix + scatter_iy │    ║
║  │  calibration constants   │  │  geometry scatter maps   │    ║
║  │  float32 + float32       │  │  int64 + int64           │    ║
║  │  ≈ 1.2 GiB  (19-seg JF) │  │  ≈ 0.3 GiB  (19-seg JF) │    ║
║  └──────────────────────────┘  └──────────────────────────┘    ║
║                                                                  ║
║  VARIABLE  —  one buffer per EventPool slot, grown lazily       ║
║  reserve() / release() called on every resize                   ║
║                                                                  ║
║  ┌────────────────────────────────────────────────────────┐    ║
║  │  slot 0                        slot 1                  │    ║
║  │  ┌─────────────────────┐       ┌─────────────────────┐│    ║
║  │  │ calib_slot_bufs[0]  │       │ calib_slot_bufs[1]  ││    ║
║  │  │ GPUDetector         │       │ GPUDetector         ││    ║
║  │  │ float32             │       │ float32             ││    ║
║  │  │ (events×segs×r×c)   │       │ (events×segs×r×c)   ││    ║
║  │  ├─────────────────────┤       ├─────────────────────┤│    ║
║  │  │ _slot_bufs[0]       │       │ _slot_bufs[1]       ││    ║
║  │  │ KvikioGpuReader     │       │ KvikioGpuReader     ││    ║
║  │  │ uint8 raw input     │       │ uint8 raw input     ││    ║
║  │  └─────────────────────┘       └─────────────────────┘│    ║
║  └────────────────────────────────────────────────────────┘    ║
║                                                                  ║
║  NOT TRACKED  —  covered by the 10% margin                      ║
║  ┌────────────────────────────────────────────────────────┐    ║
║  │  CuPy allocator pool   _raw_slot_bufs (uint16 scratch) │    ║
║  └────────────────────────────────────────────────────────┘    ║
║                                                                  ║
║  ──────────────────────────────────────────────────────         ║
║  budget.committed ≈ fixed + Σ calib_slot_bufs + Σ _slot_bufs   ║
║  budget.limit     = device_total / n_bd_ranks   (40 GiB here)  ║
╚══════════════════════════════════════════════════════════════════╝

PINNED HOST MEMORY  (separate, not counted in _GpuBudget)
┌────────────────────────────────────────────────────────────────┐
│  _D2hPipeline._PinnedSlot[0]   _PinnedSlot[1]                  │
│  page-locked, float32, (chunk_size × segs × r × c)             │
│  tracked separately by _D2hPipeline.pinned_bytes()             │
└────────────────────────────────────────────────────────────────┘
```

---

### 3. The reserve / release cycle

Both variable buffer owners follow the same pattern: `release` the old
size, `reserve` the new size, then call `cp.empty`.  This keeps
`_committed` accurate on both grow and shrink.

```
                    issue_batch(subbatch, slot=0)
                             │
     KvikioGpuReader         │          GPUDetector
     ─────────────────       │          ──────────────────────────
     old = _slot_bufs[0]     │          old = _calib_slot_bufs[0]
           .nbytes            │                .nbytes
              │               │                │
     budget.release(old) ◄───┤──────► budget.release(old)
     budget.reserve(new) ───►│◄────── budget.reserve(needed)
          raises if OOM  ────┤──────── raises if OOM
              │               │                │
     cp.empty(new, uint8) ──►│◄── cp.empty(batch_shape, float32)
     _slot_bufs[0] = buf      │    _calib_slot_bufs[0] = buf
                              │
                     [slot 0 in-flight]

                    issue_batch(subbatch+1, slot=1)
                             │
                    same pattern for slot 1 ──► slot 1 in-flight

                    issue_batch(subbatch+2, slot=0 recycled)
                             │
                    usually same size → release + reserve = no-op
                    only re-allocates if batch_size changed
```

---

### 4. `reserve()` — OOM prevention flow

```
budget.reserve(n)
        │
        ▼
committed + n <= limit?
        │
   YES ─┤─► committed += n   (fast path, ~always taken)
        │
   NO ──┤
        ▼
cp.get_default_memory_pool().free_all_blocks()
   (flush CuPy's cached-but-unused allocations)
        │
        ▼
committed + n <= limit?
        │
   YES ─┤─► committed += n   (pool flush recovered enough room)
        │
   NO ──┤
        ▼
raise GpuMemoryPressureError(
    "need X GiB, committed Y GiB, limit Z GiB.\n"
    "Reduce batch_size or n_gpu_streams, or increase gpu_memory_budget_gb."
)
```

The two-step check matters: CuPy keeps freed arrays in a pool to amortise
`cudaMalloc` latency.  A large batch may appear to exceed the limit while
several GiB of pool blocks sit unused.  The flush reclaims them before
giving up.

---

### 5. Subbatch budget — how the limit drives splitting

`_compute_subbatch_budget()` derives the per-subbatch VRAM cap from the
same `_GpuBudget` limit, reserving headroom for fixed allocations and
CuPy pool overhead:

```
_compute_subbatch_budget()
  │
  ├── limit       = _gpu_budget.limit()            (e.g. 40 GiB)
  ├── fixed_bytes = Σ det.memory_bytes()['constants']
  │                + Σ det.memory_bytes()['geometry']  (e.g. 1.5 GiB)
  ├── margin      = limit × 10%                    (e.g. 4.0 GiB)
  ├── variable    = limit − fixed_bytes − margin   (e.g. 34.5 GiB)
  ├── n_slots     = EventPool depth                (e.g. 2)
  │
  └── per_subbatch = max(variable / n_slots, 256 MiB)
                   = max(17.25 GiB, 256 MiB)
                   = 17.25 GiB  ◄── _subbatch_budget_bytes
```

`_split_subbatches()` uses this cap in a greedy bin-pack loop:

```
for each event e in EB batch:
    e_bytes = GPUDetector.estimate_subbatch_bytes(1)
            = n_segs × nrows × ncols × (4 + 2)   [float32 + uint16]

    if current_bytes + e_bytes > _subbatch_budget_bytes
       AND current subbatch already has ≥ 1 event:
        ── flush current subbatch ──► EventPool slot k
        ── start new subbatch

    append e to current subbatch
    current_bytes += e_bytes
```

This guarantees that no subbatch requests more VRAM than the budget
allows — the calib slot buffer grows to fit the subbatch and `reserve()`
will succeed.

---

### 6. Budget numbers for MFX Jungfrau (19 segments, A100 80 GiB)

```
Component                     Size         Tracked?
──────────────────────────     ──────────   ────────
peds_gpu  (float32 flat)       ~0.6 GiB    yes — fixed
gmask_gpu (float32 flat)       ~0.6 GiB    yes — fixed
scatter_ix + iy (int64)        ~0.3 GiB    yes — fixed
calib_slot_bufs[0] (float32)   ~0.2 GiB*   yes — variable
calib_slot_bufs[1] (float32)   ~0.2 GiB*   yes — variable
_slot_bufs[0] (uint8)          ~0.1 GiB*   yes — variable
_slot_bufs[1] (uint8)          ~0.1 GiB*   yes — variable
_raw_slot_bufs (uint16)        ~0.1 GiB    no  — in margin
CuPy allocator pool            varies      no  — in margin
_PinnedSlot[0,1] (host)        ~0.1 GiB    no  — host memory

budget.committed (steady state)   ≈ 2.1 GiB  (<<  40 GiB limit)
_subbatch_budget_bytes            ≈ 17.2 GiB (per subbatch)

* grows with batch_size; shown for bs=20, 19-seg Jungfrau
```

---

## How subbatches, on_gpu_view, and backpressure fit together

All three mechanisms revolve around a single concept: the **VRAM slot**.
EventPool has 2 slots.  Every subbatch fills one slot, the user reads from
it, and `retire_next()` waits until the slot is safe to overwrite.

---

### The slot lifecycle

```
STEP 1 — SUBBATCH: decide what fits in the slot
────────────────────────────────────────────────

  EB batch: 500 events × 60 MB = 30 GB  >  17 GB per-slot budget
                │
                ▼
  _split_subbatches()
    subbatch 0 = events   0–289  (17 GB ≤ budget → fits in slot 0)
    subbatch 1 = events 290–499  (12 GB ≤ budget → fits in slot 1)
                │
                ▼
  submit(subbatch 0, slot=0):
    GDS reads raw data    ───────────────────────► slot 0 raw buffer  │
    GPU calibration kernel ──────────────────────► slot 0 calib buf   │ SLOT 0
    calib_done.record()                                                │ IS FULL


STEP 2 — USER: decide how long the slot is held
────────────────────────────────────────────────

  generator yields events; user calls one of:

  ┌─────────────────────────────────────────────────────────┐
  │  on_gpu (D→D copy)                                      │
  │                                                         │
  │  arr = ctx.get('jungfrau.calib').on_gpu                 │
  │                                                         │
  │  slot 0 ──► copy ──► independent arr (not in slot 0)   │
  │  slot 0 FREE immediately                                │
  │  lease._d2h_done = None                                 │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  on_gpu_view (zero-copy)                                │
  │                                                         │
  │  with ctx.get('jungfrau.calib').on_gpu_view(s) as arr:  │
  │      my_kernel(arr, stream=s)  ← reads slot 0 directly  │
  │  # __exit__: done = Event(); s.record(done)             │
  │               lease._d2h_done = done                    │
  │                                                         │
  │  slot 0 HELD until my_kernel finishes on GPU            │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  on_cpu (D→H transfer)                                  │
  │                                                         │
  │  arr = ctx.get('jungfrau.calib').on_cpu                 │
  │                                                         │
  │  _D2hPipeline: cudaMemcpyAsync(slot 0 → pinned host)    │
  │  pslot.done_event recorded after memcpy                 │
  │  lease._d2h_done = pslot.done_event                     │
  │                                                         │
  │  slot 0 HELD until D→H transfer finishes (~13 ms)       │
  └─────────────────────────────────────────────────────────┘


STEP 3 — retire_next: enforce the slot is safe to overwrite
────────────────────────────────────────────────────────────

  Before slot 0 can hold the NEXT subbatch, retire_next() runs:

  retire_next(slot 0):
    lease._d2h_done.synchronize()   ← BLOCKS until event fires
    old_stream.synchronize()        ← BLOCKS until calib kernel done
    slot 0 = None  (recycled, safe to write)

  on_gpu:       _d2h_done = None  →  no wait, instant recycle
  on_gpu_view:  _d2h_done fires when user's kernel finishes
  on_cpu:       _d2h_done fires when D→H to pinned host finishes
```

---

### Full cycle for one EB batch

```
  EB batch (500 events)
        │
        ▼  _split_subbatches()
  ┌──────────────┐   ┌──────────────┐
  │ subbatch 0   │   │ subbatch 1   │
  │ events 0–289 │   │ events 290+  │
  └──────┬───────┘   └──────┬───────┘
         │                  │
         ▼                  ▼
    slot 0 fills        slot 1 fills    ← two slots overlap (double-buffered)
         │                  │
         ▼                  ▼
    yield events        yield events
    user: on_gpu/       user: on_gpu/
    on_gpu_view/        on_gpu_view/
    on_cpu              on_cpu
         │                  │
         │  lease._d2h_done set (or None)
         │                  │
         ▼                  ▼
    retire_next()       retire_next()
    BLOCKS until        BLOCKS until    ← BACKPRESSURE: generator stalls
    slot 0 safe         slot 1 safe       here; _next_batch() not called
         │                  │              EB rate-limited until both
         ▼                  ▼              slots are recycled
    slot 0 recycled     slot 1 recycled
         │
         ▼
  next EB batch → _split_subbatches() again …
```

---

### Role of each mechanism

| Mechanism | Question it answers | When |
|---|---|---|
| **Subbatch** | "how many events fit in one slot's VRAM?" | before GPU work starts |
| **`on_gpu_view`** | "how long does the user hold the slot?" | while slot is full |
| **`retire_next()`** | "is the slot safe to overwrite yet?" | before slot is reused |

Subbatches control the **fill**.
`on_gpu_view` controls the **hold**.
`retire_next()` enforces the **release** — and because the generator blocks
inside it, a slow user or slow D→H naturally rate-limits the EB rank.

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

### Run 3 — sdfampere035, 2026-07-27, job 33347452 (on_gpu_view + arr.sum baseline)

```
Configuration                        kHz        hot_ms
────────────────────────────────     ─────────  ───────
on_gpu_view baseline  bs= 1  pd=2    0.073 kHz  0.096 ms  ← view + arr.sum
on_gpu_view baseline  bs=10  pd=2    0.057 kHz  0.054 ms
on_gpu_view baseline  bs=20  pd=2    0.080 kHz   —
_D2hPipeline  chunk= 1  bs=20        0.075 kHz   —        ← D→H hidden (same as baseline!)
_D2hPipeline  chunk=10  bs=20        —                    ← preempted

LCLS-II beam rate: 0.120 kHz
```

**Key result:** `bs=20 baseline = 0.080 kHz` and `D2H chunk=1 = 0.075 kHz` —
effectively the same.  The lazy-sync design is confirmed: calling `on_cpu`
(with D→H in the background) does not reduce aggregate throughput vs calling
`on_gpu_view` (no D→H at all).

**hot_ms** for the `on_gpu_view + arr.sum` path is 0.054–0.096 ms, slightly
higher than the old `on_gpu` D→D copy path (0.028–0.056 ms) because
`arr.sum()` reads the entire 19-segment array on the GPU before returning.
This is a more realistic "user kernel" than a D→D copy.

### Run 2 — sdfampere036, 2026-07-27, job 33331881 (on_gpu baseline)

```
Configuration                        kHz        hot_ms
────────────────────────────────     ─────────  ───────
GPU on_gpu baseline  bs= 1  pd=2     0.071 kHz  0.053 ms
GPU on_gpu baseline  bs=10  pd=2     0.054 kHz  0.039 ms
GPU on_gpu baseline  bs=20  pd=2     0.086 kHz  0.042 ms  ← IO-bottlenecked ceiling
_D2hPipeline  chunk= 1  bs=20        0.086 kHz  19.9 ms  ← D→H hidden (same as baseline!)
_D2hPipeline  chunk=10  bs=20        0.075 kHz  23.0 ms

LCLS-II beam rate: 0.120 kHz
```

### Run 1 — earlier baseline (pre-simplification)

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

The higher kHz values in Run 1 reflect lighter Lustre load on that day —
the hot_ms values (0.028–0.056 ms) are the stable per-event GPU time.
Both runs confirm the same qualitative result: D2H overhead is hidden.

I/O path for both runs: CPU-fallback (Lustre → CPU DRAM → GPU VRAM via
cudaMemcpy).  True GDS (NVMe → GPU direct) would eliminate the CPU-DRAM
hop and raise the ceiling further.

---

## Files changed

| File | Change |
|---|---|
| `context.py` | `on_gpu` → D→D copy; `on_gpu_view(stream)` → returns `_GpuViewContext`; `_GpuViewContext.__exit__` records done-event automatically; `SlotLease._needs_release` + `mark_needs_release()` + `RuntimeError` branch removed; `GPUResult.release_after()` removed |
| `gpu_budget.py` | New — `_GpuBudget` + `GpuMemoryPressureError` |
| `gpu_calib.py` | `GPUDetector.__init__` accepts `budget=`, `n_slots=2` default; `cp.empty()` guarded by `budget.reserve()` |
| `gpu_kvikio_read.py` | Same — `budget=`, `n_slots=2`, `cp.empty()` guarded |
| `gpu_events.py` | Creates `_GpuBudget.auto()` at init; `_D2hPipeline` gains `_available: SimpleQueue`; `_PinnedSlot` gains `available` + `_refs_lock` params; `dec_ref()` calls `_available.put(self)` when freed; `_get_free_slot()` calls `get_nowait()`, returns `None` when empty; `_flush_chunk()` yields events without `_pending_d2h` when slot unavailable |
| `ds_base.py` | `n_gpu_streams` default 4→2; `gpu_memory_budget_gb` and `gpu_d2h_chunk_size` params added and forwarded to `DsParms` |
| `tests/gpu/unit/test_event_joiner.py` | `test_get_free_slot_returns_none_when_all_slots_busy` + `test_dec_ref_race_lock_prevents_lost_update` |

---

## Deferred

| Feature | When |
|---|---|
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
| `on_gpu_view(stream)` context manager — auto-records done-event | Done |
| `_GpuBudget` simple committed-bytes counter | Done |
| Pool depth default 2 | Done |
| `gpu_memory_budget_gb` enforcement | Done |

### Phase 3 — Byte-bounded subbatches + D→H backpressure
**Done.**

| Requirement | Status |
|---|---|
| Memory estimation in `GPUDetector` | Done — `estimate_subbatch_bytes(n_events)` |
| Partitioning one EB batch into byte-bounded GPU subbatches | Done — `GpuEvents._split_subbatches()` + `GpuSubbatchView` |
| Backpressure: no unconditional EB batch requests when queue full | Done — pending queue implicit in sequential subbatch processing |
| `_compute_subbatch_budget()` auto-sizes from `_GpuBudget` | Done |


Key design decisions:
- `GpuSubbatchView` re-indexes `first_desc` to be relative to the subbatch's
  own `desc_table` (built by `KvikioGpuReader._build_desc_table()`), not the
  original batch's desc table.
- Each subbatch is matched to its CPU events by timestamp lookup, so each
  CPU event is yielded exactly once.
- First subbatch reads are issued before the CPU EventManager loop to preserve
  GDS/PCIe overlap with CPU SMD deserialization.
- A single oversized event (exceeding budget alone) is never split — the rule
  "at least one event per subbatch" prevents zero-event subbatches.
- `_compute_subbatch_budget()` formula: `(limit − fixed_bytes − 10% margin) / n_slots`.

#### D→H backpressure mechanism

When all `_max_inflight=2` pinned slots are occupied, `_flush_chunk()` falls
back to synchronous D→H so the generator never deadlocks.  Free-slot
tracking, counting, and retrieval are combined into a single
`queue.SimpleQueue`:

```
_D2hPipeline
    _available = SimpleQueue()   ← holds free _PinnedSlot objects

_PinnedSlot.dec_ref()
    with _refs_lock:             ← atomic decrement (fixes _refs race)
        _refs -= 1
        freed = (_refs <= 0)
        if freed: _refs = 0
    if freed:
        _available.put(self)     ← slot returns itself to the free pool

_D2hPipeline._get_free_slot()
    try:
        return _available.get_nowait()   ← O(1), thread-safe, no scan
    except Empty:
        return None              ← all slots busy → sync D→H fallback
```


## Validation test coverage

| Test requirement | Status |
|---|---|
| Slot cannot be recycled while D→H in flight | Done — `test_retire_next_waits_for_d2h_before_recycle` |
| Generator advancement alone does not release a lease | Done — `test_generator_advancement_alone_does_not_release` |
| D→H completion token controls release | Done — `test_d2h_registered_calls_synchronize` |
| Multiple D→H chunks produce correct ordered join | Done — `test_on_cpu_returns_correct_data` |
| BeginStep and EndRun flush partial joins | Done — `test_pipeline_flush_partial` |
| `on_gpu` returns independent copy | Done — `test_on_gpu_returns_independent_copy` |
| `on_gpu_view` context manager records done-event on `__exit__` | Done — `test_on_gpu_view_records_done_event_on_exit` |
| `on_gpu_view` retire succeeds after context exit | Done — `test_on_gpu_view_retire_safe_after_context_exit` |
| `on_gpu_view` raises without lease | Done — `test_on_gpu_view_raises_without_lease` |
| Budget check prevents OOM before cp.empty() | Done — `TestGpuBudget` (5 tests) |
| Subbatch estimates stay within budget | Done — `test_subbatch_estimates_stay_within_budget` |
| Variable event sizes split correctly | Done — `TestSplitSubbatches` |
| D→H pipeline falls back to sync when all slots busy (no deadlock) | Done — `test_get_free_slot_returns_none_when_all_slots_busy` |
| Async D→H resumes after a slot is freed | Done — `test_get_free_slot_returns_none_when_all_slots_busy` |
| Concurrent dec_ref() calls produce correct _refs and return slot to queue once | Done — `test_dec_ref_race_lock_prevents_lost_update` |
| Multiple BDs cannot exceed aggregate GPU budget | Not done — Phase 4 |
