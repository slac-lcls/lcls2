# Lazy-Sync D→H Design

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

### 3. `GPUResult.on_cpu` — three-path property

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

## Timeline: one chunk of 10 events

```
Time →   0ms       10ms      20ms      30ms      40ms      50ms
         │          │         │         │          │          │
GPU:     [── calib events 0-9 on stream_0 ──────────────────►]
                    │
             calib_done ← CUDA event

_flush_chunk() at ~10ms:
         stream.wait_event(calib_done)
         memcpyAsync(pinned, view) ─────────────────────────►[done ~50ms]
         done_event.record()
         yield ctx_0 .. ctx_9  ← all 10 yielded at ~10ms


user calls on_cpu(ctx_0) at ~12ms:
         _pending_d2h.get()
           done_event.synchronize() ───────── blocks until ~50ms ────────►
           pslot.arr[0].copy()    ← ~0.1ms
           return numpy array     ← at ~50ms

user calls on_cpu(ctx_1..9) at ~50ms:
         done_event already fired → synchronize() returns instantly
         pslot.arr[i].copy()     ← ~0.1ms each
```

**The GPU starts calibrating the next batch at ~10ms** (as soon as
`_flush_chunk()` yields).  The user's first `on_cpu` call pays the
~40 ms D→H cost.  All subsequent calls in the same chunk are instant.

---

## What this changes vs the blocking design

| | Blocking (old) | Lazy-sync (new) |
|---|---|---|
| When does generator wait? | Inside `_yield_ready()` | Never |
| When does user wait? | Never (on_cpu returns instantly) | At first `on_cpu` call |
| GPU idle during D→H? | Yes | No |
| chunk=1 throughput | 65 evt/s | 72 evt/s |
| chunk=10 throughput | 53 evt/s | 73 evt/s |
| chunk size matters for throughput? | Yes (larger = worse) | No (all equal) |

---

## Batch-boundary flush

If `batch_size % chunk_size != 0` some events are left in `_chunk_buf`
at the end of a batch.  Without action they would not be yielded until
the next batch fills them to `chunk_size`, adding a full batch period
of latency.

`_yield_ready()` calls `_flush_d2h_pipelines()` after every batch.
This flushes any partial chunk immediately, so no event is stranded:

```
batch_size=15, chunk_size=10

  events 0-9  → chunk full → D→H issued → yield immediately
  events 10-14 → partial chunk (5 events)
  end of batch → _flush_d2h_pipelines()
                   └─ pipe.flush() → D→H issued for 5 events → yield
```

---

## Implementation phase status

From `gpu_memory_backpressure_and_async_join.md`.

### Phase 0 — Measurement and accounting
**Not done.**
- No per-owner byte counters
- No high-water marks for raw input, gather, output, CuPy pool, or pinned memory
- No validation of estimates against CUDA memory information

### Phase 1 — Correct slot ownership
**Done.**
- `SlotLease` carries `calib_done` CUDA event and `_d2h_done` token
- `EventPool.submit()` records `calib_done` after all calibration kernels and creates one lease per event
- `EventPool.retire_next()` calls `lease.wait_until_safe_to_reuse()` before synchronising — generator advancement alone no longer recycles a slot
- `_PendingD2H.get()` calls `dec_ref()` to signal the slot is safe to reuse after D→H

### Phase 2 — Bounded asynchronous D→H
**Mostly done.**

| Requirement | Status |
|---|---|
| Direct async D→H from slot view to pinned chunk | Done — `_D2hPipeline._flush_chunk()` |
| Separate logical `join_size` from physical chunk bytes | Done — `gpu_d2h_chunk_size` independent of event count |
| Partial-tail flush at EndRun / BeginStep | Done — `_flush_d2h_pipelines()` called at both |
| Partial-tail flush at batch boundary | Done — called at end of every `_yield_ready()` |
| No full-size D2D join buffer | Done — copies direct from slot view |
| Lazy sync — D→H overlaps with user processing | Done — `_pending_d2h` / `on_cpu` |
| `d2h_max_inflight` byte ceiling | Partial — fixed at 2 slots; no configurable byte limit |
| `gpu_pinned_memory_budget_bytes` enforcement | Not done |
| `release_after(downstream_done)` for user GPU code | Not done |

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
| BeginStep and EndRun flush partial joins | Done — `test_pipeline_flush_partial`, `test_beginstep_flushes_before_calib_update` |
| Subbatch estimates stay within budget | Not done — Phase 3 not implemented |
| Variable event sizes split correctly | Not done — Phase 3 not implemented |
| Multiple BDs cannot exceed aggregate GPU budget | Not done — Phase 4 not implemented |
