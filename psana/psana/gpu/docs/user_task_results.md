# Batched GPU task results

The experiment/run GPU path accepts a host-only `GpuTask` declaration. Psana
invokes its function once per nonempty **selected execution subbatch**, after
requested input preparation. Memory admission may split one EventBuilder batch;
max-events tails are shorter. With `gpu_fn`, the default `batch_size` is one.
Set it explicitly to obtain batching, for example `batch_size=20`.

```python
from psana import DataSource
from psana.gpu import GpuTask


def counts(batch, stream):
    import cupy as cp
    out = cp.empty((batch.size,), dtype=cp.uint32)
    batch.publish("event_count", out)  # retain before launching work
    out.fill(1)  # callback's current stream is the supplied producer stream


ds = DataSource(
    exp="mfx100848724", run=51,
    gpu_det="jungfrau", gpu_fn=GpuTask(counts),
    batch_size=20, gpu_d2h_pinned_bytes=64 << 20,
)
for run in ds.runs():
    for evt in run.events():
        count = evt.gpu.get("event_count").on_cpu  # scalar NumPy ndarray
```

Only explicitly published names are delivered. Lookup is exact: `event_count`
does not acquire a detector prefix. No task means no automatic calibration or
output copies. A task which publishes nothing causes no output allocation or
copy. Device accessors on task outputs are unavailable in this prototype;
`.on_cpu` waits if necessary, then caches an independent NumPy result. Reading it
does not invoke the callback or initiate the normal device-to-host transfer.
Parsed input fields remain available during event delivery through
`evt.gpu.detector(name).field(algorithm, field)` and their leases.

Declare dense inputs as `GpuTask(fn, inputs=["jungfrau.raw"])`; inside the
callback, `batch.input("jungfrau.raw")` has leading event dimension N and
`batch.present("jungfrau.raw")` tracks presence. These are borrowed read-only
inputs. Generic field selectors and requested calibration constants follow the
[task contract](proposals/user_gpu_pipeline.md) and its accepted
[batched-scheduling amendment](proposals/user_kernel_batched_scheduling_20260927.md).
Device identity/field metadata
is prepared once on first use during the callback; dense-only callbacks incur
no task metadata upload. Context methods expire when the callback returns.
All GPU work must use the supplied stream. Register scratch owners with
`batch.keepalive(...)` before launch. Published buffers must not be reused or
modified before their terminal copy completes; keepalive alone is not a reuse
signal. User device allocation/size/failure policy belongs to the user.

`batch.publish(name, array)` maps the leading axis to all N selected events.
`event_indices=[...]` maps rows to explicit host integer indices in the selected
subbatch. Scalars use shape `(M,)`; empty rows can use `(M, 0, ...)`. Multiple
disjoint groups may share a name with different shapes/dtypes. Names are not an
advance schema; a later subbatch may publish new names/layouts. Arrays must be
native numeric, C-contiguous CuPy arrays on the producer device. Input names
are reserved, and each `(event, name)` may be published once. Retained results
keep their original metadata.

`gpu_d2h_pinned_bytes` caps psana's aggregate **output** pinned staging per BD,
across all names. The default is **64 MiB**. Full page-rounded allocations count,
including free cache blocks and buffers held by transfers or result tokens.
Zero, oversized groups or unavailable capacity select a synchronous copy into
ordinary host memory. The entire contiguous group takes that fallback; a batch
of full images may exceed the default even when one image would fit. Increasing
the cap can preserve overlap at the cost of more pinned memory. Cache fragmentation
can also force fallback because buffers are not evicted during the run.

Copies are queued once per contiguous publication group, with one terminal
completion event per execution. Slots/input owners stay protected until every
copy finishes. CPU results are independent of slot reuse. Closing the GPU event
iterator or exhausting it converts
any retained host rows to ordinary NumPy storage and releases its pinned cache.
The cap does not bound user-retained NumPy results, input metadata staging, or
user device arrays; each BD has its own cap. `gpu_d2h_chunk_size` remains retired.

For deterministic cleanup when breaking early or when loop-body code raises,
use the standard generator close protocol:

```python
from contextlib import closing

for run in ds.runs():
    with closing(run.events()) as events:
        for evt in events:
            process(evt)
            if enough_results():
                break
```

On serial GPU runs, closing a started event iterator is terminal: outstanding
executions and copies drain, task owners retire, and later event iteration is
empty. A bare `break` does not explicitly close an iterator retained elsewhere.
Loop-body exceptions require the `closing` scope for this guarantee. CPU-only
serial iteration keeps its previous behavior. MPI already connects generator
closure to GPU cleanup; closing one rank's iterator is not a collective request
to stop the entire MPI job. This protocol does not add a public `run.close()` API.
