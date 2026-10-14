# Known psana2 GPU Problems and Limitations

**Status:** Current issue register, reviewed against this branch on 2026-09-11.

This document records verified gaps between the intended architecture and the
implementation. It is not a proposal backlog: speculative interfaces belong
under `docs/proposals/`, and performance observations belong under
`docs/performance/`.

## Correctness and resource-management issues

### Multi-EventBuilder GPU ownership

**Impact:** high for `PS_EB_NODES > 1` when BD processes from more than one EB
group share a node or GPU.

`MPIDataSource` derives GPU identity from `bd_rank - 1`, where `bd_rank` is
local to one EB group's `bd_comm`. `is_calib_leader()`,
`bd_ranks_sharing_gpu()`, and calibration CUDA-IPC exchange use that same
per-group communicator. With multiple EB groups, rank numbering restarts, so
separate groups can select the same device, elect duplicate calibration
leaders, and each compute a budget using only its own peers.

The fix should introduce node-wide BD identity and coordination before CuPy is
imported:

- Assign devices using a node-local index over all BD processes, independent
  of EB-group rank numbering.
- Form per-device BD peer groups for one calibration owner and CUDA-IPC
  exchange.
- Divide the automatic memory budget by all BD processes on that physical
  device.
- Validate more than one EB group on a node, including uneven BD/GPU counts.

Restricting the design to `PS_EB_NODES=1` would hide the ownership problem and
is not the intended resolution. Until node-wide coordination is implemented,
multi-EB GPU placement is not a validated configuration.

Relevant code: `psexp/mpi_ds.py` and `gpu/gpu_mpi.py`.

### Result-lease fan-out

**Impact:** high when the same `GPUResult` is consumed zero-copy on more than
one CUDA stream.

`SlotLease` stores one `_consumer_done` event. Each
`on_gpu_view(stream).__exit__` assigns that field, so a second consumer
replaces the first event instead of adding another dependency. EventPool can
therefore reuse the slot after the last registered event while an earlier
consumer is still running. `InputSlotLease` already uses the required list of
completion events for parsed fields.

Until this is fixed, use at most one `on_gpu_view()` consumer stream for each
detector result. If several kernels consume the view, enqueue them on that one
stream inside one context. An independent `on_gpu` copy is safe only under the
documented default/null-stream usage; the property does not enforce that
stream itself.

The result lease should collect every registered event, wait for all of them,
and have unit coverage matching `InputSlotLease`. `GPUResult.on_gpu` should
also either force its copy onto the null stream or explicitly register the
actual copy stream.

Relevant code: `gpu/context.py`, `gpu/gpu_input.py`, and `gpu/gpu_stream.py`.

### Fixed-allocation accounting

**Impact:** high under tight VRAM limits or when several BD processes share a
GPU.

`_GpuBudget` reserves KvikIO input buffers, parser tables, and detector slot
buffers. Calibration constants are allocated by `prep_calib_constants()` and
geometry maps by `prepare_geometry_from_arrays()` without reserving their
bytes. `_compute_subbatch_budget()` subtracts the measured fixed bytes while
deriving its default per-slot estimate, but `_GpuBudget.committed()` still
under-reports actual live allocation and cannot reject those allocations
before CUDA does.

The budget needs explicit fixed-allocation ownership. Reservation must occur
before allocation, followers using CUDA-IPC views must not reserve the
leader-owned bytes again, and replacement/cleanup must release the correct
amount. Tests should compare category totals, committed bytes, and mocked CUDA
allocation behavior for leaders and followers.

Relevant code: `gpu/gpu_budget.py`, `gpu/gpu_calib.py`,
`gpu/gpu_detector.py`, and `gpu/gpu_events.py`.

### Subbatch admission is an estimate, not a hard fit guarantee

**Impact:** medium for unusually small budgets or a single oversized event.

The default subbatch allowance has a 256 MiB floor, even if the calculated
available per-slot memory is smaller. Splitting also always admits the first
event of a subbatch when that event alone exceeds the estimate. Allocation-time
`_GpuBudget.reserve()` is therefore the final guard and can still reject work
after splitting.

The implementation contains a `gpu_subbatch_budget_bytes` attribute lookup,
but `DsParms` does not expose that as a supported `DataSource` argument. Either
make the override a validated public/internal configuration or remove the dead
configuration path. A future admission check should report an oversized
single event before beginning slot allocation.

Relevant code: `gpu/gpu_events.py` and `psexp/ds_base.py`.

## Incomplete pipeline behavior

### Automatic D2H covers calibrated dense results only

`GpuEventManager` creates `_D2hPipeline` only for `<det>.calib`. The pipeline
assumes a dense three-dimensional float32 result. Raw detector results,
arbitrary parser fields, image results, and proposed user-task outputs are not
automatically staged to host memory.

Also, `_is_fully_host_backed()` refuses early slot release whenever parsed
input dgrams are attached. Current GPU events expose parser-backed fields
eagerly, so automatic calibrated-result D2H normally overlaps the copy but does
not make the whole event host-only before yield.

Pinned memory is count-bounded by `max(2, n_gpu_streams)` slots per detector
pipeline, but has no explicit byte cap. Its allocation scales with detector
result size and `gpu_d2h_chunk_size`. Generalizing this path requires declared
output shape/dtype, a host-byte budget, and a policy for which published
results receive a host handoff. Those requirements also apply to the
[user GPU pipeline proposal](proposals/user_gpu_pipeline.md).

Relevant code: `_D2hPipeline`, `_is_fully_host_backed()`, and
`_retire_issue_and_yield()` in `gpu/gpu_events.py`.

### GPU `RunParallel.steps()` is not implemented

On a GPU BD rank, `RunParallel.steps()` returns without yielding. BeginStep is
handled only while iterating `run.events()`, where the manager drains dependent
work and refreshes calibration constants. GPU applications that require the
public step iterator need a unified step-envelope implementation rather than a
second GPU event path.

Relevant code: `RunParallel.steps()` in `psexp/mpi_ds.py`.

### Detector image results are not published

Geometry upload and `GPUDetector.assemble_image()` exist, and `EventContext`
has an `image_gpu` field. `GPUDetector.process_batch()` never calls the helper
or assigns that field, so normal processing publishes `<det>.raw` and
`<det>.calib` but not `<det>.image`.

Before enabling images, decide whether assembly is always-on, explicitly
requested, or a user-pipeline stage. Include its output allocation in
subbatch admission and define D2H policy independently from calibrated data.

Relevant code: `gpu/gpu_calib.py`, `gpu/gpu_detector.py`, and
`gpu/gpu_stream.py`.

### `smd_callback` cannot be combined with GPU routing

Callback batching produces CPU and step batches but not the GPUBAT1 descriptor
packet. `DsParms` rejects `smd_callback` together with `gpu_det` or
`hybrid_det`. Supporting the combination requires callback filtering to keep
the CPU and GPU packets coherent for exactly the same selected events.

Relevant code: `psexp/ds_base.py` and EventBuilder batch construction.

## Closure standard

An item should leave this document only after the behavior is implemented,
covered by a focused unit or integration test, and reflected in the current
design documents. Experimental measurements alone do not close a correctness
or ownership issue.
