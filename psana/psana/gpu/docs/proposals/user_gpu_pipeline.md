# User GPU Pipeline

**Status:** Proposed for team review; no public API in this document is
implemented or committed.

## Goal

Psana should stage detector data and calibration metadata in GPU memory, invoke
user GPU tasks inside its internal BD batch pipeline, and manage declared task
buffers, completion events, asynchronous host delivery, and slot reuse. Psana
should not define or interpret the user's GPU algorithm.

The defining property is placement. User GPU work is submitted before the
public event is yielded, not from the external `run.events()` loop:

```text
GPU read and XTC parsing
  -> stage detector inputs and calibration constants
  -> invoke user tasks on the slot stream
  -> record one producer-completion event
  -> optionally schedule bounded asynchronous D2H
  -> publish GPUResult objects
  -> yield the public Event
```

This lets GPU reads and kernels overlap CPU event construction and work on
another EventPool slot. Merely wrapping a function called from the user's event
loop would provide packaging convenience but would not remove the producer
synchronization boundary.

The current implementation has no `GpuTask`, task registry, task context, or
`gpu_pipeline` DataSource argument. It always calls
`GPUDetector.process_batch()` after parsing. The APIs below describe the
desired boundary for replacing or extending that fixed detector-processing
stage.

## Intended user model

A compiled CUDA task receives a task context and psana's non-blocking slot
stream. The following is illustrative API, not a finalized ABI:

```cpp
#include "jungfrau.hh"
#include "threshold.hh"
#include <psana_gpu_task.hpp>

extern "C" int userfunc(void* task_context, void* user_params,
                        void* stream_handle)
{
    psana::GpuTaskContext evt(task_context);
    auto stream = static_cast<cudaStream_t>(stream_handle);
    auto* params = static_cast<MyParams*>(user_params);

    auto raw = evt.input<uint16_t>("jungfrau.raw");
    auto peds = evt.calibconst<float>("jungfrau", "pedestals");
    auto gain_mask = evt.calibconst<float>("jungfrau", "gain_mask");

    auto calib = evt.scratch<float>("jungfrau.calib");
    auto output = evt.output<float>("jungfrau.threshold");

    jungfrau_calib<<<grid, block, 0, stream>>>(
        raw.data(), peds.data(), gain_mask.data(), calib.data(), raw.size());
    threshold<<<grid, block, 0, stream>>>(
        calib.data(), output.data(), params->threshold, raw.size());
    return 0;
}
```

A Python/CuPy adapter should have the same data, buffer, stream, and lifetime
semantics:

```python
def userfunc(evt, params, stream):
    raw = evt.input("jungfrau.raw")
    peds = evt.calibconst("jungfrau", "pedestals")
    gain_mask = evt.calibconst("jungfrau", "gain_mask")
    calib = evt.scratch("jungfrau.calib")
    output = evt.output("jungfrau.threshold")

    with stream:
        jungfrau_calib(raw, peds, gain_mask, calib)
        threshold(calib, output, params.threshold)
```

Both adapters represent one task contract. A separate process-wide kernel
registry, `schedule_for()` graph, and ordinary callback list are not required
for the first implementation.

## Pipeline configuration

Inputs, scratch space, and outputs must be declared before processing so psana
can validate names and include every managed allocation in subbatch admission:

```python
task = GpuTask(
    function="libuser.so:userfunc",  # or a Python/CuPy callable
    params={"threshold": 5.0},
    inputs=["jungfrau.raw"],
    scratch={
        "jungfrau.calib": ("float32", "same"),
    },
    outputs={
        "jungfrau.threshold": ("float32", "same"),
        "jungfrau.nhits": ("uint32", "scalar"),
    },
)

ds = DataSource(
    exp="mfx100848724",
    run=51,
    gpu_det="jungfrau",
    gpu_pipeline=[task],
)
```

The exact spelling of `GpuTask` and `gpu_pipeline` remains open. The important
contract is declarative buffer ownership, not the constructor syntax.

Tasks form an ordered pipeline. A later task may consume a named output from an
earlier task. The first version should use declared list order and reject an
input that has no detector producer or earlier task producer. A general kernel
registry or dependency DAG can be considered later if real use cases require
it.

## Psana and user responsibilities

Psana owns:

- Detector stream routing, bigdata reads, XTC parsing, and event identity.
- Stable device views for declared detector inputs and calibration constants.
- Per-`(slot, name)` scratch and output arenas with one event slice per task
  invocation.
- Byte-budget admission for all psana-managed inputs, scratch, and outputs.
- Task invocation on the slot stream and the final completion event.
- Optional pinned-host D2H staging and published `GPUResult` lifetimes.

The task owns:

- Kernel code, launch geometry, scalar parameters, and algorithm semantics.
- Correct use of the supplied CUDA stream.
- Writes to every element it publishes.
- Explicit ordering for persistent state shared across slot streams.
- Any persistent allocation not declared to psana.

Tasks must not perform a blocking D2H, synchronize the device, or allocate
per-event output with `cudaMalloc`/`cupy.empty`. They may perform one-time setup
such as compilation or plan creation, but an eventual lifecycle interface is
preferable to detecting the first event inside the hot path.

## Multiple outputs and scratch

`output(name)` and `scratch(name)` solve the multiple-pointer problem without
embedding algorithm-specific fields in the task context:

- Output buffers are named, published, and accessible after the task.
- Scratch buffers are named but remain internal to the task chain.
- Shape and dtype declarations let psana size the subbatch before submission.
- Each event receives non-overlapping slices from reusable per-slot arenas.
- Recycled memory is undefined on entry; a task must fully initialize outputs.

Shape specifications should initially support `same`, `scalar`, explicit
integer tuples, and dimensions derived from the first input. Variable-capacity
outputs such as peak lists need a declared maximum-capacity array plus a
device-side count; host-dependent allocation in the producer loop would break
asynchronous execution.

## Completion and asynchronous D2H

The task must not call D2H itself. Psana records a CUDA event after the final
task on the slot stream. A D2H stream waits on that event and copies selected
outputs into bounded pinned-host slots with `cudaMemcpyAsync` or the equivalent
CuPy operation.

```text
slot stream:  read -> parse -> task 0 -> task 1 -> producer_done
                                                        |
D2H stream:                              wait(producer_done) -> host copy -> host_done
```

The public loop remains conventional:

```python
for evt in run.events():
    process_cpu_data(evt)
    threshold = evt.gpu.get("jungfrau.threshold").on_cpu
```

`on_cpu` waits for that result's host-completion event only when the CPU first
touches it; it must not globally synchronize the GPU. If automatic D2H is
disabled, the existing blocking, cached one-result copy remains available.

Device consumption uses the existing result rules:

- `on_gpu` creates an independent D2D copy.
- `on_gpu_view(stream)` exposes a zero-copy slot view and registers the
  downstream stream's completion with the slot lease.
- A slot is reusable only after its producer, automatic D2H, and registered GPU
  consumers have all completed.

That last rule is a requirement for this proposal, not yet a complete property
of `SlotLease`: the current detector-result lease stores only one downstream
event. A user-task implementation must collect all terminal consumers (as the
current parsed-input `InputSlotLease` does) before it can safely fan one output
out to multiple streams.

## Calibration placement

The example deliberately lets the user task consume raw data and calibration
constants and replace the built-in Jungfrau calibration launch. This should be
an explicit pipeline choice rather than an implicit side effect of registering
a callback.

One possible long-term model represents the current Jungfrau calibration as a
built-in task in the same ordered pipeline. The default pipeline would preserve
today's `raw` and `calib` results. Image assembly could become an explicit
built-in stage; geometry helpers exist today, but normal processing does not
publish an `image` result. An explicitly supplied pipeline could replace or
extend the built-in stages. The first implementation must define this
replacement rule before stabilizing configuration names.

Calibration pointers are step-scoped. Psana must drain dependent task work
before replacing them at `BeginStep`. A task must fetch current constants for
each invocation unless a future lifecycle callback provides an explicit safe
refresh point.

## Internal execution sketch

```python
for subbatch in gpu_batch.split_to_budget():
    slot = event_pool.acquire()
    reads = reader.issue_batch(subbatch, slot)
    cpu_events = event_manager.materialize_cpu_events(subbatch)

    reader.finish(reads)
    parsed = xtc_parser.parse(slot, subbatch)

    for event_input in parsed.events():
        for task in configured_pipeline:
            task.enqueue(event_input, slot.buffers, slot.stream)

    slot.producer_done.record(slot.stream)
    d2h_pipeline.enqueue_declared_outputs(slot)
    publish_gpu_results(cpu_events, slot)
```

This is ordering pseudocode, not a prescription to block on every read or to
materialize Python objects in the hot loop. Implementations should preserve the
existing batched descriptor and locator tables.

## Concurrency constraints

Tasks for one event and slot run in declared order on that slot's stream.
Different EventPool slots use different streams and may overlap. A task that
updates run-wide device state must therefore provide its own cross-stream
ordering or use one partial state allocation per slot and reduce later.

CUDA Graphs are a possible optimization, not part of the initial contract.
Stable per-slot addresses and declared maximum capacities make graph capture
possible, but partial batches, dynamic shapes, and stateful analysis require
explicit policies. The callback itself is host code; only captured GPU work is
replayed.

## Error handling

- Missing detector data may skip a task for that event without publishing its
  outputs.
- Undeclared names, dtype mismatches, budget failures, and task error returns
  fail fast with the task name and event timestamp.
- C++ exceptions must not cross a C ABI boundary.
- On failure, psana must account for already-enqueued work before releasing or
  reusing a slot.

## Open decisions before implementation

1. Whether the stable compiled entry point includes an explicit parameters
   pointer or exposes parameters through the task context.
2. The lifecycle for compile/setup, `BeginStep`, and teardown state.
3. How an explicitly configured pipeline replaces or extends built-in detector
   calibration.
4. Which outputs receive automatic D2H and how that policy is configured.
5. Whether v1 invokes once per event or also offers a batch-task ABI to reduce
   Python and launch overhead.
6. The minimum C ABI and C++ wrapper surface shared with the Python/CuPy
   adapter.

These decisions should be resolved with a small end-to-end threshold prototype
before adding registries, automatic dependency graphs, or CUDA Graph APIs.
