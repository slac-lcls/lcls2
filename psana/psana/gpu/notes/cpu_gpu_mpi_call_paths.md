# CPU and GPU MPI Call Paths

## Status

This note describes the implemented MPI paths on
`codex/psana2-gpu-two-phase-retire` at commit
`67e4a08249c829d96c8a8b5e3e4d347deeaed100`.

The active source remains authoritative. In particular, the MPI GPU transition
path has a known gap described below: transitions reach the GPU-owning BD
process inside the SMD packet, but the MPI adapter currently supplies an empty
`step_dict` to `GpuEvents`.

## Terminology

```text
SMD batch
    The normal EventBuilder CPU batch. It contains CPU-routed L1Accept data and
    non-L1 transitions. GPU-routed stream entries are omitted from L1Accept
    events in GPU mode.

GPUBAT1
    A versioned descriptor batch for GPU-routed L1Accept data. It contains
    event identity, stream IDs, bigdata offsets, and read sizes, but no detector
    payload and no transition payload.

step_batch
    An EventBuilder-local copy of non-L1 transitions. EventBuilderNode uses it
    to maintain per-BD StepHistory. It is not a third EB-to-BD wire packet.
```

The EB-to-BD message always has two outer packets:

```text
CPU mode: [repacked SMD events + transitions] [empty GPU packet]
GPU mode: [repacked CPU-SMD events + transitions] [GPUBAT1]
```

`repack_for_bd()` prepends transitions that the selected BD missed because a
different BD received the batch in which those transitions originally
appeared.

## CPU vs. GPU MPI Call Path

| Stage | CPU path | GPU path | Short description |
|---|---|---|---|
| User setup | `DataSource(exp=..., run=...)` | `DataSource(..., gpu_det="jungfrau")` | The same factory is used; `gpu_det` enables GPU stream routing. |
| DataSource selection | `DataSource()` -> `MPIDataSource()` | Same | Constructs MPI communicators and the role-specific DataSource. |
| GPU assignment | No GPU action | `MPIDataSource.__init__()` -> `init_gpu_rank()` | Only BD ranks receive a GPU; SMD0 and EB ranks have CUDA visibility disabled. |
| Run creation | `MPIDataSource.runs()` -> `RunParallel()` | Same | Every SMD0, EB, and BD rank constructs its role-specific `RunParallel`. |
| Role setup | `RunParallel.__init__()` -> `Smd0`, `EventBuilderNode`, or `BigDataNode` | Same | The MPI topology determines the role. |
| SMD0 read | `RunParallel.start()` -> `Smd0.start()` | Same | GPU mode does not change SMD0; it distributes normal SMD chunks. |
| EB construction | `EventBuilderNode.start()` -> `EventBuilderManager()` | Same | Receives SMD chunks and constructs timestamp-aligned event batches. |
| EventBuilder routing | `gpu_stream_ids=None` | `gpu_stream_ids=[...]` | Configure metadata identifies streams carrying the requested GPU detector. |
| EventBuilder build | `EventBuilder.build()` -> `_build_fast_batch()` | `EventBuilder.build()` -> `_build_fast_batch_gpu_split()` | CPU builds a normal SMD batch. GPU builds a CPU SMD batch plus GPUBAT1 descriptors. |
| Transition copy | Transition is placed in `smd_batch` and `step_batch` | Same | `step_batch` updates EB `StepHistory`; transitions themselves travel in the SMD packet. |
| EB output | `repacked_smd + empty GPU packet` | `repacked_smd + GPUBAT1` | `step_batch` is not sent as a third packet. |
| BD event-loop fork | `RunParallel.events()` -> `self.start()` | `RunParallel.events()` -> `_gpu_events_mpi()` | This is the principal CPU/GPU BD fork. |
| BD receive | `RunParallel.start()` -> `BigDataNode.start()` | `_gpu_events_mpi()` -> `self.start()` -> `BigDataNode.start_gpu()` | CPU receives through `Events`; GPU receives raw SMD/GPUBAT1 pairs. |
| MPI adaptation | None | `_MpiGpuBatchSource` -> `_OneBatch.next_with_gpu()` | Adapts the flat MPI batch stream to the serial-style interface expected by `GpuEvents`. |
| Batch controller | `Events.__next__()` | `GpuEvents._events()` | CPU delegates to `EventManager`; GPU coordinates CPU and GPU work. |
| Descriptor parsing | `EventManager._get_offset_and_size()` | `GpuBatchView()` -> `_split_subbatches()` | CPU parses offsets from SMD dgrams; GPU parses validated fixed-width descriptors. |
| Bigdata read | `EventManager._fill_bd_chunk()` -> `os.pread()` | `KvikioGpuReader.issue_batch()` -> `CuFile.pread()` | CPU reads into host buffers. GPU targets a reusable GPU buffer through GDS or KvikIO CPU fallback. |
| Read completion | Retried `os.pread()` | `KvikioGpuReader.wait_batch()` -> `future.get()` | Waits for reads and verifies byte counts. |
| CPU event creation | `EventManager` -> `Event(dgrams=...)` | Same `EventManager` for CPU-routed streams | The GPU detector's dgram is absent from the CPU Event in GPU mode. |
| Detector processing | User calls `det.raw.calib(evt)` | `EventPool.submit()` -> `GPUDetector.process_batch()` | CPU detector calibration is generally user-triggered; GPU calibration runs before result delivery. |
| Raw extraction | Detector-specific CPU code | `GPUDetector._extract_and_calibrate()` | Locates raw panels and gathers multi-segment GPU data. |
| Calibration | CPU detector implementation | `fused_calib_gpu()` | Applies Jungfrau pedestal, gain, and mask processing in a CUDA kernel. |
| CPU/GPU correlation | Not needed | Timestamp lookup in `GpuEvents` | Correlates the CPU Event with the GPU result from the coherent batch. |
| Result ownership | Ordinary detector result | `_EventSlot` + `SlotLease` | Protects reusable GPU storage until terminal consumers complete. |
| Optional D2H | CPU data is already on host | `_D2hPipeline.schedule()` -> `memcpyAsync()` | Transfers calibrated output to bounded pinned-host buffers. |
| User event type | `Event` | `GpuEventContext` | GPU context combines a normal CPU Event with GPU result handles. |
| User access | `det.raw.calib(evt)` | `ctx.get("calib").on_gpu`, `.on_gpu_view()`, or `.on_cpu` | Selects an independent GPU copy, leased GPU view, or CPU result. |
| Transition handling | `Run._handle_transition()` | Intended: `GpuEvents._handle_steps()` -> `_dispatch_transition()` | GPU work must drain before BeginStep constants change. The MPI adapter currently does not provide the step dictionary. |
| Shutdown | `Events`/`EventManager` exhaustion | `_flush_event_pool()` -> `_drain_pending_gpu_read()` -> `gpu_reader.close()` | GPU shutdown drains consumers and any pre-issued read before closing. |

## Shared SMD0 and EventBuilder Front Half

The CPU and GPU MPI paths share the same SMD0 transport and EventBuilder event
alignment. The difference begins when Configure-derived `gpu_stream_ids` enable
the EventBuilder GPU split.

```text
DataSource()
  -> MPIDataSource()
  -> MPIDataSource.runs()
  -> RunParallel()

SMD0 rank:
  RunParallel.start()
    -> Smd0.start()
    -> SmdReaderManager
    -> SmdReader / ParallelReader
    -> normal SMD chunks

EB rank:
  RunParallel.start()
    -> EventBuilderNode.start()
    -> EventBuilderManager.batches_with_gpu()
    -> EventBuilder.build()
       CPU: _build_fast_batch()
       GPU: _build_fast_batch_gpu_split()
    -> repack_for_bd()
    -> pack(repacked_smd, gpubat1_or_empty)
    -> MPI send to selected BD
```

For a non-L1 transition, both EventBuilder paths copy the transition into the
normal SMD batch and into `step_batch`. `EventBuilderNode` sends the transition
inside the SMD packet and uses `step_batch` to update `StepHistory` for BDs that
must receive it later.

## Condensed CPU MPI Call Chain

```text
RunParallel.events()
  -> RunParallel.start()
  -> BigDataNode.start()
     -> get_smd()
        -> send BD request to EB
        -> receive two-packet EB message
        -> _unpack_batch()
        -> return repacked SMD packet
     -> Events
     -> EventManager
        -> _get_offset_and_size()
        -> _get_bd_offset_and_size()
        -> _fill_bd_chunk()
        -> os.pread()
        -> _get_next_dgrams()
     -> yield dgrams
  -> RunParallel._handle_transition(dgrams)
     -> update EnvStore and swallow non-L1 transitions
  -> Event(dgrams=..., run=...)
  -> yield Event
  -> user: det.raw.raw(evt) / det.raw.calib(evt)
```

On the CPU path, `BigDataNode.start()` returns already-materialized dgram lists.
`RunParallel.events()` is therefore a thin generator that handles transitions,
constructs `Event`, and yields it to user code.

## Condensed GPU MPI Call Chain

```text
RunParallel.events()
  -> RunParallel._gpu_events_mpi()
     -> BigDataNode.start_gpu()
        -> send initial BD request to EB
        -> receive two-packet EB message
        -> _unpack_batch()
           -> repacked SMD packet
           -> GPUBAT1 bytes
        -> send look-ahead request for the next batch
        -> yield smd_batch, gpubat1_bytes
     -> _MpiGpuBatchSource
        -> _OneBatch.next_with_gpu()
        -> batch_dict, gpu_batch_dict, step_dict
     -> GpuEvents._events()
        -> GpuBatchView()
        -> _split_subbatches()
        -> KvikioGpuReader.issue_batch()
        -> EventManager() for CPU-routed streams
        -> KvikioGpuReader.wait_batch()
        -> EventPool.submit()
           -> GPUDetector.process_batch()
           -> GPUDetector._extract_and_calibrate()
           -> fused_calib_gpu()
           -> record result_ready
           -> create SlotLease objects
        -> optional _D2hPipeline.schedule()
        -> correlate CPU events and GPU results by timestamp
        -> _yield_ready()
        -> GpuEventContext
     -> yield GpuEventContext
  -> user: ctx.get("calib")
       -> .on_gpu
       -> .on_gpu_view(stream)
       -> .on_cpu
```

`BigDataNode.start_gpu()` performs transport, outer-packet unpacking,
look-ahead, termination, and batch statistics. It does not parse GPUBAT1 or
perform CUDA work. `_MpiGpuBatchSource` is a compatibility adapter because
`GpuEvents` expects the serial two-level `SmdReaderManager -> BatchIterator ->
next_with_gpu()` interface, while MPI already supplies one completed
EventBuilder batch per message.

## GPU Result Lifetime

The GPU execution path is batch-oriented even though it yields one
`GpuEventContext` per event:

```text
KvikIO input slot
  -> GPUDetector calibration-output slot
  -> EventPool _EventSlot
  -> result_ready CUDA event
  -> terminal consumer
     automatic D2H or user GPU operation
  -> consumer completion event
  -> slot reuse
```

With `gpu_d2h_chunk_size=0`, users may request an independent D2D copy with
`on_gpu` or register a zero-copy consumer with `on_gpu_view(stream)`. With
automatic D2H enabled, the normal steady-state path transfers the result to
pinned host memory, releases the device slot, and exposes the result through
`on_cpu`.

## MPI GPU Transition Gap

The intended GPU transition path is:

```text
step_dict
  -> GpuEvents._handle_steps()
  -> flush EventPool at BeginStep or EndRun
  -> GpuEvents._dispatch_transition()
  -> GPUDetector.beginstep() for BeginStep
  -> Run._handle_transition()
```

This works when the batch source supplies the real EventBuilder `step_dict`, as
in the serial GPU path. In the current MPI GPU path,
`_MpiGpuBatchSource._OneBatch.next_with_gpu()` returns an empty `step_dict`.
Transitions still arrive inside the repacked SMD packet, but
`GpuEvents._events()` currently skips non-L1 dgrams in its `EventManager` loop
without dispatching them. Consequently, MPI GPU BeginStep calibration refresh
and transition-driven EnvStore updates are not connected in this snapshot.

## Source Map

- `psana/psana/psexp/mpi_ds.py`: `RunParallel.events()` and
  `_gpu_events_mpi()`.
- `psana/psana/psexp/node.py`: `Smd0`, `EventBuilderNode`, `StepHistory`,
  `repack_for_bd()`, and `BigDataNode`.
- `psana/psana/psexp/smdreader_manager.py`: serial `BatchIterator` and
  `next_with_gpu()` interface.
- `psana/psana/eventbuilder.pyx`: normal and GPU-split EventBuilder paths.
- `psana/psana/gpu/gpu_batch.py`: GPUBAT1 ABI and subbatch views.
- `psana/psana/gpu/gpu_events.py`: GPU orchestration and CPU/GPU timestamp
  correlation.
- `psana/psana/gpu/gpu_kvikio_read.py`: KvikIO/cuFile bigdata reads.
- `psana/psana/gpu/gpu_stream.py`: EventPool and reusable-slot lifetime.
- `psana/psana/gpu/gpu_calib.py`: raw extraction and Jungfrau calibration.
- `psana/psana/gpu/context.py`: `GpuEventContext` and `GPUResult`.
