# Limits and open work

Reviewed against code at `fa40ec52a`, 2026-09-29, plus the GPU batch-default
update to 20. Completed fixes belong in the
current design and tests; this page contains current restrictions and open work.

## Multi-EB device accounting

**Open correctness/resource issue:** BD GPU pinning and automatic peer counts
use the EB-local `bd_comm`. With several EB groups on one node, local rank
numbering restarts: groups can select the same GPU while budgeting only their
own peers. `PS_EB_NODES > 1` on shared GPUs is not a validated configuration.
The accepted campaigns use one EB. Resolving this requires node-wide BD device
assignment and memory accounting before CUDA initialization, followed by uneven
multi-EB/BD/GPU placement tests. An explicit memory quota alone does not implement
that coordination. Sources: [mpi_ds.py](../../psexp/mpi_ds.py),
[gpu_mpi.py](../gpu_mpi.py).

## Supported interface and configuration

| Area | Current boundary |
| --- | --- |
| DataSource mode | `gpu_fn` supports experiment/run serial and MPI input; files-only, shmem and DRP modes are rejected |
| GPU step iteration | Use `run.events()` for transitions; GPU `RunParallel.steps()` returns without yielding |
| SMD callback | `smd_callback` with GPU routing is rejected because callback batching does not create coherent GPUBAT1 descriptors |
| Bulk reads | `gpu_bulk_read=True` rejects `intg_det` and nonempty timestamp filtering; disabling bulk removes that parameter restriction, not a claim of acceptance for those combinations |
| Stream routing | Exclusive routing requires one normal detector per selected stream; hybrid routing duplicates complete-stream I/O |
| Dense inputs | Public adapter is Jungfrau raw uint16; generic parsed fields are available without a dense adapter |
| Task outputs | Named `.on_cpu` results; no task-output device view/copy API |
| Scheduling default | 20 with GPU routing, with or without a task; CPU-only remains 1000; explicit values override |
| Retired interface | Bare callable `gpu_fn` and nonzero `gpu_d2h_chunk_size` are rejected |

Source validation is in [ds_base.py](../../psexp/ds_base.py), with public entry
points in [run.py](../../psexp/run.py) and [mpi_ds.py](../../psexp/mpi_ds.py).

## Memory and cleanup contracts

The framework device ledger does not bound arbitrary user scratch/output,
independent copies or CUDA/KvikIO allocations. The output pinned cap excludes
input/task metadata staging and retained NumPy arrays. User allocations therefore
need their own memory policy, particularly when many BDs share one GPU. Oversized
output groups use blocking ordinary-host copies, which can change performance.

Borrowed inputs/constants are read-only by contract; native kernel writes cannot
be intercepted. A saved raw view is not safe after its lease ends. Register
scratch/output owners before launching work and use the supplied stream.
There is no public output-reuse notification for user buffer pools.

A bare `break` does not close an iterator retained elsewhere. Use
`contextlib.closing(run.events())`; MPI local close is not collective termination.
Fatal pipeline errors abort MPI. See [lifetime and error handling](design.md#transitions-close-and-errors).

## Scientific and measurement scope

External Jungfrau calibration and fixed-bin radial integration are implemented
and validated for their stated policies. Common-mode correction, solid-angle,
polarization and pixel splitting are outside these examples. A fixed radial map
is not automatically a run-specific q-space calibration.

The current performance evidence uses KvikIO CPU fallback on A100 hardware.
True-GDS correctness/performance and broader detector/topology acceptance remain
separate work. Batching benefits are workload-dependent; the current reports
must not be interpreted as a universal no-regression guarantee. The GPU default
of 20 follows the latest JF staging and user-kernel scaling settings. Good scaling
requires investigating batch size for the kernel work, scratch/output memory and
GPU/BD layout; other values still need workload-specific validation. The default
does not account for arbitrary user scratch/output memory.
