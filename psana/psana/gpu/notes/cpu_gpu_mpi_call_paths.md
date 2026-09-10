# CPU and GPU MPI event paths

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
| GPU setup | None | `_make_gpu_event_manager()` | Creates one run-scoped GPU manager, uploads Configure-derived XTC tables, and shares calibration through CUDA IPC. |
| Run dispatch | `RunParallel.start(None)` | `RunParallel.start(manager)` | Passes the optional processor into the common BD path. |
| MPI receive | `BigDataNode._batch_envelopes()` | Same | Receives the two-packet EB message and posts one-batch look-ahead. |
| Transport value | `BatchEnvelope(smd, None)` | `BatchEnvelope(smd, gpubat1)` | Keeps the coherent CPU/GPU communication unit together. |
| Stream controller | `Events.__next__()` | Same | Requests another batch only after the active event-envelope iterator is exhausted. |
| GPU read issue | None | `KvikioGpuReader.issue_batch()` | Starts reads from GPUBAT1 bigdata descriptors into the selected slot's VRAM buffer. |
| CPU materialization | `EventManager` | `EventManager` inside `GpuEventManager` | Reads CPU bigdata and constructs `EventEnvelope(dgrams)`. |
| GPU XTC parse | None | `GpuXtcBatchPool.parse()` | Uploads dgram records, walks XTC, and locates registered array fields on the slot stream. |
| GPU detector | None | `GPUDetector.process_batch()` | Uses Configure-selected handles and device locator rows to produce canonical raw, calibrated, and optional image results. |
| Internal result | `EventEnvelope(dgrams)` | `EventEnvelope(dgrams, gpu_state)` | Carries one event without owning RunCtx. |
| Public result | `RunParallel` creates `Event(gpu=None)` | `RunParallel` creates `Event(gpu=GpuEventState)` | The same public object is returned in both modes. |
| User GPU access | N/A | `evt.gpu.get("calib")` | Returns a lease-aware `GPUResult`. |

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
       submit detector kernels on the same stream
       correlate CPU and GPU records by timestamp
       attach GpuEventState to each EventEnvelope
  -> Events
  -> RunParallel.events()
       Event(envelope.dgrams, run=RunCtx, gpu=envelope.gpu_state)
  -> yield Event
```

`GpuEventManager` is run-scoped. It owns the CPU `GpuStreamConfigTable`, the
`GpuXtcBatchPool`, KvikIO reader, GPU detectors, D2H pipelines, shared VRAM
budget, and EventPool. `GpuXtcBatchPool` uploads its numeric Configure tables
once and owns reusable parser buffers indexed by EventPool slot. EventPool
retains each subbatch's `GpuEventBatch` until that slot is safely retired.

`GpuEventState` is event-scoped and contains only that event's detector
results, leases, pending D2H tokens, and cached host results. Stage 2 does not
expose XTC dgram or field-locator tables through the public Event API.

Stage 2 runs the GPU XTC parser in shadow mode: it produces device dgram and
field-locator tables before detector processing, but `GPUDetector` continues
to use legacy raw-array addressing until the Stage 3 consumer switch.

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

The manager preserves the two-phase retirement window:

```text
begin_retire_next()
yield Event to user code
user registers a downstream CUDA completion event
finish_retire_next()
reuse slot
```

Advancing the Python generator is not treated as proof that an asynchronous
GPU consumer completed. `evt.gpu.get(...).on_gpu_view(stream)` records the
consumer completion token used by EventPool.

## Transitions

The SMD packet in `BatchEnvelope` already contains transition and missing-step
history. The GPU manager drains prior work before BeginStep or EndRun and
refreshes step-dependent GPU calibration through its transition handler. CPU
MPI transitions continue through `Run._handle_transition()` and are swallowed
from the public `run.events()` stream.

GPU `RunParallel.steps()` remains outside this first unification pass.
