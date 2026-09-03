# GPU Callbacks: Event-Loop Comparison And CUDA Graphs

## Status

Proposed design, not yet implemented.

This note compares ordinary per-event GPU callbacks with calling the same
Python functions directly in `run.events()`. It also identifies the narrower
producer-side scheduling hook that could provide a performance advantage and
the additional restrictions required for CUDA Graphs.

## Current Execution Model

For this example:

```text
batch_size = 5
pool_depth = 2

slot 0: batch 0, events 0..4, reusable input/calib/work buffers
slot 1: batch 1, events 5..9, reusable input/calib/work buffers
```

`batch_size` is the EventBuilder communication size. A batch may be split into
smaller GPU subbatches when required by the execution-memory limit, so the
example assumes that all five events fit one slot. `pool_depth` bounds
concurrent execution slots, not total GPU bytes.

Each slot owns a CUDA stream and reusable input, raw-gather, calibration-output,
and scratch capacity. A slot is safe to reuse only after every kernel, copy, or
other consumer reading its buffers has completed. Callback-created output and
workspace allocations are user-owned and are not bounded by `pool_depth`.

## Ordinary Callback Interface

An ordinary callback is a Python function invoked once per event:

```python
class UserAnalysis:
    def __init__(self):
        self.peaks = []

    def peak_finding(self, ctx, stream):
        with ctx.get("jungfrau.calib").on_gpu_view(stream) as calib:
            positions, energies = find_peaks(calib, stream=stream)

        # positions and energies own independent allocations. The user chooses
        # whether and how long to retain them.
        self.peaks.append((ctx.timestamp, positions, energies))

    def max_projection(self, ctx, stream):
        update_max_projection(ctx, stream=stream)


analysis = UserAnalysis()

ds = DataSource(
    exp="mfx...",
    run=123,
    gpu_det="jungfrau",
    gpu_d2h_chunk_size=0,
    gpu_callbacks=[analysis.peak_finding, analysis.max_projection],
    pool_depth=2,
)

for ctx in next(ds.runs()).events():
    # Event-loop GPU access remains available under the retained-slot contract.
    with ctx.get("jungfrau.raw").on_gpu_view(user_stream) as raw:
        consume_raw(raw, stream=user_stream)
```

Callbacks run in list order for each event. Psana does not infer dependencies,
allocate callback workspace, retain callback return values, or manage the
user's CuPy objects. A callback may put results in its own list, accumulator,
writer, or other application-owned object. Keeping a CuPy object keeps its
allocation live and may eventually produce CUDA OOM; that is user memory
policy rather than psana output-queue policy.

Psana therefore adds no output-retention container or output-capacity
DataSource argument in this model.

## Similarities With Event-Loop Calls

The equivalent explicit event-loop code is:

```python
analysis = UserAnalysis()

for ctx in run.events():
    analysis.peak_finding(ctx, analysis_stream)
    analysis.max_projection(ctx, analysis_stream)
```

The two forms share the important behavior:

- Both call ordinary Python once per event.
- CUDA launches inside either function are asynchronous with respect to
  Python.
- Both may use CuPy operations, RawKernel launches, helper functions, or a
  user-created CUDA Graph.
- The user owns workspace, persistent state, and output allocations.
- Psana cannot infer hidden data dependencies or safely parallelize arbitrary
  functions.
- The GPU slot must remain protected until every asynchronous consumer that
  reads slot-backed raw or calibrated data has completed.

For these semantics, `gpu_callbacks=[...]` is primarily configuration and
convenience. Merely moving the same per-event Python calls into psana does not
create more GPU concurrency, reduce kernel launch count, or make the work
CUDA-Graph compatible.

## Differences

| Property | Event-loop call | Ordinary `gpu_callbacks` |
| --- | --- | --- |
| Placement | Explicit in user loop | Configured on `DataSource` |
| Invocation | User chooses when and whether | Psana invokes each callback per event |
| Ordering | Written directly by user | Deterministic list order per event |
| Outputs | User stores objects | User callback stores objects |
| Dependencies | User-managed | User-managed |
| Workspace | User-managed | User-managed |
| Basic scheduling | During yielded-event window | Normally the same window |

Callbacks may still be useful for packaging reusable analysis, applying the
same analysis in serial and MPI execution, or keeping the event loop focused
on CPU analysis. Those are usability benefits, not automatic GPU-performance
benefits.

## Slot Lifetime And Event-Loop Access

Ordinary callbacks should preserve the current external-GPU retirement window:

```text
begin_retire(slot)
  -> keep slot leased
  -> for each event in the retired subbatch
       invoke configured callbacks
       yield the event
       allow event-loop on_gpu/on_gpu_view consumers
  -> wait for every registered CUDA consumer
  -> release slot
```

With this contract and `gpu_d2h_chunk_size=0`, callbacks do not cause early
slot release. Event-loop code may still request `jungfrau.raw` or
`jungfrau.calib` through `on_gpu` or `on_gpu_view`:

- `on_gpu` creates an independent D2D copy while the source slot is valid.
- `on_gpu_view(stream)` exposes a zero-copy slot view and records a completion
  event for work enqueued on `stream` inside the context manager.

Safety requires psana to wait for all callback and event-loop consumers. The
current `SlotLease` stores one consumer-completion event. Supporting a callback
and a later event-loop view on different streams therefore requires either a
collection/aggregate of completion events or explicit stream ordering that
makes the final recorded event cover all earlier work. A later registration
must not simply overwrite an unrelated outstanding consumer event.

Automatic D2H remains a different mode. With `gpu_d2h_chunk_size > 0`, psana
copies calibrated results asynchronously into a bounded pinned-host pool and
may release the device slot before yielding the event. The resulting context
supports `on_cpu`, while event-loop `on_gpu` and `on_gpu_view` remain
unavailable. Automatic D2H currently covers `<det>.calib`, not `<det>.raw`.
An implementation should either reject ordinary GPU callbacks in this mode or
run callbacks that need slot-backed data on the producer side before D2H and
wait for both consumers before reuse.

User outputs retained beyond the leased-event window must own independent
storage. A callback must not save a bare pointer or zero-copy view into a
reusable psana slot.

## One Possible Scheduling Advantage

The current external event-loop path synchronizes the producer stream before
yielding slot-backed results. A callback invoked only after that synchronization
has no scheduling advantage over a function called directly in the event loop:

```text
calibration -> producer synchronize -> callback/event-loop launches
```

A narrower producer-side implementation could invoke callbacks while the slot
is submitted, queueing their GPU work directly after calibration on the slot's
stream:

```text
slot stream: read -> calibration -> callback launches -> result_ready
CPU:                                      independent CPU work
```

This could provide three benefits:

- No host synchronization between calibration and callback kernel launches.
- Callback work can overlap CPU event construction and work in another slot.
- One final completion event can cover calibration and same-stream callback
  work.

This advantage is not inherent in a Python callback list; it depends on where
psana invokes the callbacks and which stream they use. Per-event Python calls
and kernel-launch overhead still remain. It should be measured against the
event-loop baseline before adding framework complexity.

Producer-side enqueue does not require early slot release. Psana can enqueue
callbacks before synchronization and still retain the slot through the later
event-loop access window. Releasing before yield would be a separate,
incompatible mode for contexts that need slot-backed GPU access.

## Cross-Slot Stateful Analysis

List order defines dependencies only within one event. With two slot streams,
batch 0 and batch 1 may execute concurrently. For example, both
`max_projection` callbacks must not update one run-wide array concurrently
without an ordering strategy.

Safe choices include:

- Run all projection updates on one explicitly ordered stream.
- Keep one partial maximum per slot and reduce the partials later.
- Use an algorithm whose cross-stream writes are explicitly safe.

Psana should not infer which choice is correct from arbitrary Python code.

## CUDA Graphs

A Python callback is not itself captured in a CUDA Graph. It can construct a
graph during setup or launch a previously instantiated graph for each event:

```python
def graph_callback(ctx, stream):
    # Select a graph for the stable backing slot, or update graph-node inputs.
    graph_exec = prepare_graph_exec(ctx)
    graph_exec.launch(stream=stream)
```

The captured graph may contain multiple kernels, supported CuPy/library
operations, asynchronous copies, and explicit GPU dependencies. Callbacks do
not have to be single RawKernels. Python branches, object mutation, and other
host behavior run while the graph is constructed; they are not repeated by
graph replay.

Practical restrictions include:

- Capture and replay use explicit non-default streams.
- CPU/GPU synchronization and synchronous D2H are not allowed during capture.
- Input, output, and workspace addresses must remain valid for replay. A pool
  normally needs one graph per stable slot or explicit graph-node parameter
  updates when pointers change.
- Fixed batch shapes are easiest. A partial final batch needs padding/masking,
  another graph, or a non-graph fallback.
- Variable-sized peak output normally needs preallocated maximum capacity plus
  a device-side count, rather than allocating from a CPU-visible peak count.
- A Python condition such as `if evt.energy < threshold` selects whether to
  launch a graph outside capture. Per-event conditional work inside one graph
  needs device-side predication or alternate graph executables.
- Stateful operations such as a run-wide maximum projection still require
  explicit ordering across slot streams.
- A completion event recorded after graph launch must participate in the slot
  lease before psana reuses graph input buffers.

Per-event graph replay can reduce the launch overhead of a multi-kernel chain,
but still performs one graph launch per event. Capturing calibration and user
analysis once for an entire GPU subbatch could remove more host launch overhead,
but that requires a distinct batch-oriented interface with stable shapes,
buffers, workspace, and output-capacity rules. It should not be presented as
automatic optimization of arbitrary `gpu_callbacks`.

See the CuPy `Stream.begin_capture()` documentation and the NVIDIA CUDA
Programming Guide's CUDA Graphs chapter for the capture rules:

- <https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html>
- <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>

## Design Conclusion

The initial choice is:

- Keep GPU analysis in the event loop for the smallest public API; or
- Add ordinary callbacks as a convenience mechanism with the same leased-slot
  and user-memory semantics.

Neither choice requires psana to retain or cap user outputs. A future
producer-side enqueue hook may improve overlap, and a batch-oriented processor
may support efficient CUDA Graph replay, but both are narrower contracts that
should be justified by measurements rather than implied by ordinary callbacks.
