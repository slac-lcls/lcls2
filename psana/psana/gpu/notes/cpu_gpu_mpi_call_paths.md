# CPU and GPU MPI event paths

This note describes the unified MPI event path on this branch. Both CPU and
GPU runs use the same EB-to-BD transport, the same one-batch look-ahead, and
yield `psana.Event` from `RunParallel.events()`.

## Common call chain

```text
RunParallel.events()
  -> RunParallel._events_impl()
  -> RunParallel.start(gpu_manager=None | GpuEventManager)
  -> BigDataNode.start(run, gpu_manager)
  -> BigDataNode._batch_envelopes()
       receive EB message
       unpack BatchEnvelope(smd, gpu)
       request the next EB batch before yielding the current envelope
  -> Events(batch_source, run, gpu_manager)
  -> yield Event
```

There is no separate `start_gpu()`, `_gpu_events_mpi()`, or
`_MpiGpuBatchSource` path.

## Side-by-side calls

| Stage | CPU | GPU | Purpose |
|---|---|---|---|
| Public iterator | `RunParallel.events()` | Same | User-facing event generator. |
| GPU setup | None | `_make_gpu_event_manager()` | Creates one run-scoped GPU manager and shares calibration through CUDA IPC. |
| Run dispatch | `RunParallel.start(None)` | `RunParallel.start(manager)` | Passes the optional processor into the common BD path. |
| MPI receive | `BigDataNode._batch_envelopes()` | Same | Receives the two-packet EB message and posts one-batch look-ahead. |
| Transport value | `BatchEnvelope(smd, None)` | `BatchEnvelope(smd, gpubat1)` | Keeps the coherent CPU/GPU communication unit together. |
| Stream controller | `Events.__next__()` | Same | Requests another envelope only after the active batch iterator is exhausted. |
| CPU materialization | `EventManager` | `EventManager` inside `GpuEventManager` | Reads CPU bigdata and constructs dgrams. |
| GPU processing | None | `GpuEventManager.process_batch()` | Issues KvikIO reads, launches detector work, and correlates timestamps. |
| Per-event result | `Event(gpu=None)` | `Event(gpu=GpuEventState)` | The same public object is returned in both modes. |
| User GPU access | N/A | `evt.gpu.get("calib")` | Returns a lease-aware `GPUResult`. |

## CPU path

```text
BatchEnvelope.smd
  -> EventManager
  -> dgram list
  -> Event(dgrams, run=RunCtx)
  -> RunParallel handles/swallow transitions
  -> yield Event
```

## GPU path

```text
BatchEnvelope(smd, gpubat1)
  -> GpuEventManager.process_batch()
       inspect transitions from the SMD packet
       retire the next reusable slot when necessary
       issue the first GPU read before CPU EventManager work
       run EventManager for CPU-routed streams
       wait for GPU reads and submit detector kernels
       correlate CPU and GPU records by timestamp
       attach GpuEventState to each Event
  -> Events
  -> RunParallel.events()
  -> yield Event
```

`GpuEventManager` is run-scoped because streams, KvikIO buffers, detector
constants, D2H pipelines, and EventPool slots span batches. `GpuEventState`
is event-scoped and contains only that event's results, leases, pending D2H
tokens, and cached host results. It does not reference the manager.

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
