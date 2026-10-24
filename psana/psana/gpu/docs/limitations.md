# Limits and open work

Reviewed against code at `fa40ec52a`, 2026-09-29, plus the GPU batch-default
update to 20. Completed fixes belong in the
current design and tests; this page contains current restrictions and open work.

## Multi-EB device accounting

**Supported.** BD workers sharing a GPU are grouped by driver-reported
`(hostname, PCI bus id)`, so the peer count is independent of EB topology by
construction and no rank arithmetic participates. Deriving it from the
EB-local `bd_comm` instead would over-commit the device, because local rank
numbering restarts in each group: measured 2.0x at `PS_EB_NODES=2` and 3.0x at
3, with ranks on one device disagreeing about the count in uneven layouts.

Validated on A100s at `PS_EB_NODES` 1, 2 and 3, contiguous and round-robin, an
uneven 8-rank split, and across two nodes: aggregate claim 1.0 on every device.
See
[device placement and shared constants](device_placement_and_shared_constants.md).

Two items remain open.

**The automatic budget is conservative within a device.** It takes the minimum
of free memory over the ranks sharing one card, so the busiest peer's sample
sets the figure for all of them. The minimum is per device, not job-wide: a mix
of 40 GB and 80 GB cards no longer lowers the 80 GB card's ranks to the 40 GB
share. An explicit `gpu_memory_budget_gb` is checked against each device's own
capacity and peer count, and the failure verdict is then reduced job-wide, so
every rank stops together without any device being judged against another's
figures.

**MIG is not supported and not tested, and no flag makes it work.** Grouping
keys on the PCI bus id, which instances of one card share, so ranks on separate
instances are counted as peers and each one's budget is divided by however many
were merged. `PSANA_GPU_SHARE_CONSTANTS=0` avoids the IPC abort that follows
but leaves that peer count and budget wrong, so it is not a MIG configuration.
Deferred per [issue #155](https://github.com/slac-lcls/lcls2/issues/155) as
follow-up before a MIG deployment, not a blocker for the current non-MIG scope.
See
[device placement and shared constants](device_placement_and_shared_constants.md#limits)
for the detail.

Sources: [mpi_ds.py](../../psexp/mpi_ds.py),
[gpu_placement.py](../gpu_placement.py).

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
