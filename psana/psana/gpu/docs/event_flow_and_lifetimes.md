# CPU and GPU Event Flow and Lifetimes

**Status:** Current on this branch.

This note describes the unified MPI event path on this branch. Both CPU and
GPU runs use the same EB-to-BD transport, the same one-batch look-ahead, and
yield `psana.Event` from `RunParallel.events()`.

## Common call chain

```text
RunParallel.events()
  -> RunParallel._events_impl()
  -> RunParallel.start(gpu_manager=None | GpuEventManager)
  -> BigDataNode.start(gpu_manager)
  -> BigDataNode._batch_envelopes()
       receive EB message
       unpack BatchEnvelope(smd, gpu)
       request the next EB batch before yielding the current envelope
  -> Events(batch_source, gpu_manager)
  -> EventEnvelope(dgrams, gpu_state=None | GpuEventState)
  -> RunParallel materializes and yields Event
```

There is no separate `start_gpu()`, `_gpu_events_mpi()`, or
`_MpiGpuBatchSource` path.

## Side-by-side calls

| Stage | CPU | GPU | Purpose |
|---|---|---|---|
| Public iterator | `RunParallel.events()` | Same | User-facing event generator. |
| GPU setup | None | `_make_gpu_event_manager()` | Creates one run-scoped GPU manager, uploads Configure-derived XTC tables, and creates budgeted reader/parser storage. |
| Run dispatch | `RunParallel.start(None)` | `RunParallel.start(manager)` | Passes the optional processor into the common BD path. |
| MPI receive | `BigDataNode._batch_envelopes()` | Same | Receives the two-packet EB message and posts one-batch look-ahead. |
| Transport value | `BatchEnvelope(smd, None)` | `BatchEnvelope(smd, gpubat1)` | Keeps the coherent CPU/GPU communication unit together. |
| Stream controller | `Events.__next__()` | Same | Requests another batch only after the active event-envelope iterator is exhausted. |
| GPU read issue | None | `KvikioGpuReader.issue_batch()` | Starts reads from GPUBAT1 bigdata descriptors into the selected slot's VRAM buffer. |
| CPU materialization | `EventManager` | `EventManager` inside `GpuEventManager` | Reads CPU bigdata and constructs `EventEnvelope(dgrams)`. |
| GPU XTC parse | None | `GpuXtcBatchPool.parse()` | Uploads dgram records, walks XTC, and locates registered array fields on the slot stream. |
| Dense input preparation | None | `DenseInputPreparer.prepare_batch()` | Internal optional gather of validated inputs; no calibration or automatic results. |
| Internal result | `EventEnvelope(dgrams)` | `EventEnvelope(dgrams, gpu_state)` | Carries one event without owning RunCtx. |
| Public result | `RunParallel` creates `Event(gpu=None)` | `RunParallel` creates `Event(gpu=GpuEventState)` | The same public object is returned in both modes. |
| User GPU access | N/A | `evt.gpu.detector(name).field(alg, field)` | Returns a lease-aware parsed input field. |

## CPU path

```text
BatchEnvelope.smd
  -> EventManager
  -> EventEnvelope(dgrams)
  -> RunParallel handles/swallows transitions
  -> Event(dgrams, run=RunCtx)
  -> yield Event
```

## GPU path

```text
BatchEnvelope(smd, gpubat1)
  -> GpuEventManager.process_batch()
       inspect transitions from the SMD packet
       split GPU work into byte-bounded subbatches when necessary
       retire the next reusable slot when necessary
       issue the first GPU read before CPU EventManager work
       run EventManager for CPU-routed streams
       wait for the GPU read to finish
       translate read descriptors into device dgram records
       walk XTC and locate registered fields on the slot stream
       retain input owners and record execution completion
       correlate CPU and GPU records by timestamp
       attach GpuEventState to each EventEnvelope
  -> Events
  -> RunParallel.events()
       Event(envelope.dgrams, run=RunCtx, gpu=envelope.gpu_state)
  -> yield Event
```

`GpuEventManager` is run-scoped. It owns the CPU `GpuStreamConfigTable`, the
`GpuXtcBatchPool`, KvikIO reader, optional input preparers, per-BD VRAM
budget, and EventPool. `GpuXtcBatchPool` uploads its numeric Configure tables
once and owns reusable parser buffers indexed by EventPool slot. EventPool
retains each subbatch's `GpuEventBatch` until that slot is safely retired.

`GpuEventState` is event-scoped and contains that event's detector results,
result leases, pending D2H tokens, cached host results, detector bindings, and
a slot-backed parsed-input view. The parser tables remain run/slot-owned;
general field access resolves event-specific views through the retained input
binding and lease.

The GPU XTC parser produces device dgram and field-locator tables. The optional
`DenseInputPreparer` consumes them without fixed payload offsets or child order.
No calibration or output publication runs in the current input-only path.

See [GPU XTC parser](gpu_xtc_parser.md) for table layouts and parser details.

## Look-ahead

Every BD uses the same bounded request schedule:

```text
send request 0
receive envelope N
send request N+1
process envelope N
```

This overlaps EB construction with CPU bigdata work or GPU work. A BD has at
most the current envelope and one prefetched message. There is no BD startup
lineup: ranks begin requesting independently, and the EB selects among ranks
that are currently waiting.

## GPU result lifetime

Parsed input delivery preserves a two-phase retirement
window:

```text
begin_retire_next()
yield Event to user code
user registers a downstream CUDA completion event
finish_retire_next()
reuse slot
```

Advancing the Python generator is not treated as proof that an asynchronous
GPU consumer completed. `field.on_gpu_view(stream)` registers completion with
an input lease. All registered consumer streams must finish before input storage
can be reclaimed. Retained public views can keep input windows live beyond an
execution. The old image D2H path is removed; generic publication is planned in
[Stages 2–4](proposals/user_kernel_implementation_stages_20260926.md).

## Transitions

The SMD packet in `BatchEnvelope` already contains transition and missing-step
history. The GPU manager drains prior work before BeginStep or EndRun. It
dispatches host transitions without preparing GPU calibration constants. CPU
MPI transitions continue through `Run._handle_transition()` and are swallowed
from the public `run.events()` stream.

GPU `RunParallel.steps()` remains outside this first unification pass.
