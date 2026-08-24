# GPU Callbacks: Capacity-Based User Kernel Model

## Status

Proposed design, not yet implemented.

## Goal

Support arbitrary user GPU analysis without requiring psana to understand the
callback's CUDA kernels, workspace, output shapes, or CUDA graph. Keep the
framework responsibility narrow:

- Protect the reusable slot-backed `jungfrau.raw` input until all callbacks
  that read it have completed.
- Retain references to callback outputs in a bounded `output_results`
  container.
- Require the user to drain retained outputs often enough to stay within the
  configured capacity.

Psana does not byte-budget, allocate, resize, or otherwise manage user output
memory. A count limit bounds retained result records, not their VRAM usage.

## Proposed Interface

```python
ds = DataSource(
    exp="mfx...",
    run=123,
    gpu_det="jungfrau",
    gpu_d2h_chunk_size=0,
    gpu_callbacks=[peak_finding, max_projection],
    pool_depth=4,
    max_pending_gpu_outputs=1000,
)
```

The controls are independent:

| Control | Meaning |
| --- | --- |
| `pool_depth` | Maximum concurrent psana execution slots and callback work |
| `max_pending_gpu_outputs` | Maximum retained user-result records per BD |
| User drain interval | Application-selected CPU/GPU join or consumption point |

Increasing output capacity does not create more CUDA streams or execution
slots. Increasing `pool_depth` does not increase the number of outputs the
application may retain.

## Callback Semantics

A callback may contain Python conditions, CuPy operations, RawKernel launches,
calls to other functions, or a user-defined CUDA graph:

```python
def peak_finding(ctx, stream):
    if ctx.raw("gmd").energy >= 5_000:
        return None

    with ctx.get("jungfrau.raw").on_gpu_view(stream) as raw:
        positions, energies = find_peaks(raw, stream=stream)

    return {
        "positions": positions,
        "energies": energies,
    }
```

A plain `gpu_callbacks=[a, b, c]` list has deterministic, ordered semantics.
Psana invokes `a`, then `b`, then `c` for each event and queues their GPU work
on one supplied stream. It records one completion event after the complete
callback chain.

Returned outputs must be independent of the psana `jungfrau.raw` execution
slot. Psana stores the returned CuPy arrays or user result object, not raw
device pointers. Holding the Python object preserves its allocation lifetime;
it does not transfer byte-level memory management to psana.

## Output Results

Psana retains at most one composite record per event with non-`None` callback
output:

```text
GpuOutputRecord
  timestamp
  event_index
  results       callback name -> returned user object
  ready         CUDA completion event
```

`output_results` holds strong references to the returned objects until the
user drains or discards them. If the user keeps another reference after a
drain, that allocation remains live. Dropped CuPy allocations may also remain
cached in the CuPy memory pool.

`max_pending_gpu_outputs` is a capacity guard, not a byte budget. Variable or
large outputs can exhaust VRAM before the count limit is reached. Capacity
exhaustion should fail rather than block, because only the user event loop can
drain the container:

```text
GpuOutputCapacityError:
1000 pending GPU output records are retained. Drain more frequently or
increase max_pending_gpu_outputs.
```

An opaque callback or a single event may still cause CUDA OOM. Psana should
report the callback, timestamp, pending count, and available GPU-memory
diagnostics, then terminate distributed execution coherently. It cannot safely
recover or recommend only one tuning knob because OOM may come from callback
workspace, output size, persistent user state, or `pool_depth`.

## Call Path And Slot Release

```text
Run.events()
  -> GpuEvents receives one GPU subbatch
  -> EventPool acquires a free execution slot
  -> KvikIO fills the slot-backed Jungfrau raw buffer
  -> raw_ready is recorded
  -> callback stream waits for raw_ready
  -> callbacks run in configured list order
  -> callback_done is recorded
  -> non-None outputs are retained in output_results
  -> EventPool waits for or observes callback_done
  -> Jungfrau raw slot is released and may be reused
```

The important lifetime split is:

```text
jungfrau.raw slot lifetime
    ends at callback_done

user output lifetime
    ends when output_results is drained/discarded and no user references remain
```

Python callback return is not proof of GPU completion. The raw slot must remain
protected until the recorded CUDA event completes. Conversely, retained user
outputs do not keep the raw slot leased when they own independent allocations.

This early-release mode means the slot-backed `jungfrau.raw` view is valid only
inside the callback phase. Event-loop code cannot request that view after psana
has released the slot. A callback using `on_gpu` instead obtains an independent
device copy, but its copy and later outputs remain user-owned memory.

## User Example

```python
import cupy as cp
from psana import DataSource


def peak_finding(ctx, stream):
    if ctx.raw("gmd").energy >= 5_000:
        return None

    with ctx.get("jungfrau.raw").on_gpu_view(stream) as raw:
        mask = raw > 100
        positions = cp.argwhere(mask)
        energies = raw[mask]

    return positions, energies


ds = DataSource(
    exp="mfx...",
    run=123,
    gpu_det="jungfrau",
    gpu_d2h_chunk_size=0,
    gpu_callbacks=[peak_finding],
    pool_depth=4,
    max_pending_gpu_outputs=1000,
)

run = next(ds.runs())

for i, ctx in enumerate(run.events()):
    # Normal CPU detector analysis can proceed in the event loop.
    energy = ctx.raw("gmd").energy

    if (i + 1) % 1000 == 0:
        records = run.gpu.output_results.drain()
        join_cpu_and_gpu(records)

# Drain the partial tail at end of input.
records = run.gpu.output_results.drain()
if records:
    join_cpu_and_gpu(records)
```

`drain()` should return records only after their `ready` events complete, or
return explicit readiness handles while transferring object ownership to the
user. A separate `discard()` operation must defer reference release until GPU
work has completed. A plain list `clear()` is insufficient because it could
drop the final output reference while a kernel is still writing it.

## Parallel Callbacks

The initial implementation should not infer callback independence or
parallelize a plain list. Arbitrary callbacks may share writable arrays,
captured state, library state, or hidden streams, and concurrent memory-bound
kernels may be slower.

If measurements justify it, a later API may add explicit parallel groups:

```python
gpu_callbacks=[
    parallel(peak_finding, roi_sum),
    classify_peaks,
]
```

The user would assert that callbacks in the group are independent. Psana would
fan out from `raw_ready`, run each branch on its own stream, fan completion
events back into one `callback_done`, and release the raw slot only after that
aggregate event. Stateful callbacks such as a run-wide maximum projection need
ordered execution across events unless they provide their own safe concurrency.

Until such an API is needed, advanced users can encapsulate parallel streams or
a CUDA graph inside one callback and return a completion event covering all
work.

## Design Boundary

This model intentionally favors simplicity and scaling over complete memory
safety for arbitrary user code:

- Psana protects only its reusable `jungfrau.raw` source slot.
- Psana bounds retained output records by count.
- Users own callback workspace and returned GPU allocations.
- Users choose output capacity and drain frequency.
- CUDA OOM remains the failure boundary for unknown user memory demand.

This is separate from a fully managed GPU-stage model, which would require
declared inputs, outputs, workspaces, and byte-level admission control.
