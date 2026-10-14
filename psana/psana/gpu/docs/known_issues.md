# Known psana2 GPU Problems and Limitations

**Status:** Current issue register, updated after Stage 4 and lifecycle fixes on 2026-09-28.

This document records verified gaps between the intended architecture and the
implementation. It is not a proposal backlog: speculative interfaces belong
under `docs/proposals/`, and performance observations belong under
`docs/performance/`.

## Correctness and resource-management issues

### Multi-EventBuilder GPU ownership

**Impact:** high for `PS_EB_NODES > 1` when BD processes from more than one EB
group share a node or GPU.

`MPIDataSource` derives GPU identity from `bd_rank - 1`, where `bd_rank` is
local to one EB group's `bd_comm`. `bd_ranks_sharing_gpu()` uses that same
per-group communicator. With multiple EB groups, rank numbering restarts, so
separate groups can select the same device and compute a budget using only
their own peers.

The fix should introduce node-wide BD identity and coordination before CuPy is
imported:

- Assign devices using a node-local index over all BD processes, independent
  of EB-group rank numbering.
- Divide the automatic memory budget by all BD processes on that physical
  device.
- Validate more than one EB group on a node, including uneven BD/GPU counts.

Restricting the design to `PS_EB_NODES=1` would hide the ownership problem and
is not the intended resolution. Until node-wide coordination is implemented,
multi-EB GPU placement is not a validated configuration.

Relevant code: `psexp/mpi_ds.py` and `gpu/gpu_mpi.py`.

### Result-lease fan-out (fixed in ownership Stage 3)

`SlotLease` now collects every terminal event. Open zero-copy contexts pin the
slot; retirement rejects fresh access and can be retried after those contexts
exit. `GPUResult.on_gpu` records completion on the actual copy stream.
Retired facades release backing references while independent CPU caches remain
available. Escaped raw ndarray aliases remain charged but are not snapshots
and must not be used after their context ends.

See [Stage 3 ownership findings](bulk_ownership_stage3_findings.md) for the
implementation and [Stage 4 acceptance](bulk_ownership_stage4_findings.md) for
the completed single-BD JF validation.

### Accounting boundary outside pipeline-owned device storage

Admission reserves Configure tables, reader/parser buffers, and any explicitly
prepared inputs. Calibration output and geometry allocations have been removed.

The ledger covers participating owners, not every CUDA allocation in the
process. User-owned independent GPU copies, escaped array references, custom
kernel allocations, and CUDA/KvikIO runtime allocations are outside it; the
10% allocator margin is headroom, not a bound on arbitrary user allocations.
User-task outputs remain outside the device ledger. Output host staging has a
separate aggregate pinned-memory cap of 64 MiB per BD by default.

Reader/parser buffers still require their existing lifetime reservations, and
execution storage must drain all supported consumer leases before trimming.
Returning capacity to the ledger is separate from CuPy's cached free blocks.

Relevant code: `gpu/gpu_budget.py`, `gpu/gpu_admission.py`,
`gpu/gpu_events.py` and `gpu/gpu_detector.py`.

## Incomplete pipeline behavior

### User-task scope and remaining examples

Batched callbacks and named host outputs are implemented; see the
[task/results guide](user_task_results.md). There are no implicit calibrated
results. Bare callables and nonzero `gpu_d2h_chunk_size` are rejected. Task
outputs currently expose `.on_cpu` only; parsed input fields retain GPU access.
The calibration-plus-azimuthal-integration example remains pending. Historical
calibration benchmarks must be identified separately from staging-only runs.

Geometry variants absent from startup shared caches now compute per rank,
without event-loop collectives. Serial GPU iterator closure now drains resources;
use `with closing(run.events())` when breaking early or handling loop-body errors.
These fixes and their tests are recorded in the
[geometry](geometry_cache_fallback_20260928.md) and
[serial cleanup](serial_gpu_close_20260928.md) reports.

### GPU `RunParallel.steps()` is not implemented

On a GPU BD rank, `RunParallel.steps()` returns without yielding. BeginStep is
handled only while iterating `run.events()`, where the manager drains dependent
work before dispatching the host transition. GPU applications that require the
public step iterator need a unified step-envelope implementation rather than a
second GPU event path.

Relevant code: `RunParallel.steps()` in `psexp/mpi_ds.py`.

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
