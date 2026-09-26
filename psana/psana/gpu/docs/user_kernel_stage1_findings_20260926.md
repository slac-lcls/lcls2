# User-kernel Stage 1: dense input preparation

2026-09-26. Branch `codex/psana2-gpu-user-kernels`, implementation based on
`480f7074c`. This implements the internal preparation boundary from the
[stage plan](proposals/user_kernel_implementation_stages_20260926.md).
`GpuTask` and `DataSource(gpu_fn=...)` remain unimplemented.

## Implemented boundary

`DenseInputPreparer` in `gpu_detector.py` owns the existing batched gather,
raw/presence slot buffers, routing upload, and owner/row maps. It returns one
`PreparedInputBatch` per nonempty execution: original event descriptors,
event-major dense data, and per-event/per-segment device presence. It neither
accepts calibration constants nor creates calibration outputs or geometry.
There is no new scheduler, registry, or device allocation policy.

`GPUDetector` uses that preparer before applying its existing calibration.
Float32 passthrough gathers directly into its existing calibrated slot, with
no extra raw buffer or device copy. Legacy setup and BeginStep refresh are
isolated in `_setup_legacy_detector()` and `_refresh_legacy_calibration()` in
the shared serial/MPI GPU manager. Their ordering and behavior are preserved.

`DenseInputPreparer.jungfrau_raw()` establishes the supported panel layout from
the Jungfrau adapter contract and validates the Configure detector, segment,
algorithm, field, numeric type, and rank. The supported source shapes are
uint16 `(512, 1024)` and `(1, 512, 1024)`, matching the detector panel definition
and DRP writer. Configure Names alone does not encode those runtime dimensions.
The gather checks them on device without copying locator rows to the CPU.
A same-byte-count but wrong-shape payload is rejected through `present=0` and
zero-filled data. This stricter check is enabled for the new preparer; the
legacy calibration adapter keeps its prior layout checks.

Dense rows follow the binding's canonical physical segment IDs. For sparse
IDs such as `(9, 4)`, rows zero and one refer to physical segments nine and
four. User calibration will select those IDs along the unchanged dictionary
array's physical-segment axis. Raw gain bits remain intact.

## Ownership, memory, and launches

The input windows/parser tables retain their existing owners. Prepared views
borrow execution-slot storage. Callers retain input leases, order producer
streams, and wait for all registered consumer completions before slot reuse or
trimming. The new component does not infer release from Python iteration.

Raw/presence/routing/map allocations use the existing byte budget and allocation
owners. Estimates and growth requirements exclude calibrated outputs and
constants for raw-only preparation. Trimming drops cache references; backing
charges survive any retained array aliases. Pinned row-map storage remains
reported separately.

There is still one gather launch per nonempty detector execution subbatch.
Shape checks run in that gather, adding no launch or locator D2H. Binding/table
resolution occurs at setup. No per-field wrapper/launch loop was introduced.
Default calibration still launches its existing per-event calibration and
missing-row cleanup. No throughput improvement is claimed.

## Validation

Validation snapshots use current Python sources and packaged data plus the
verified Integrated native binaries inherited from the master-merge snapshot.
No native extension sources changed. Manifests record source hashes; test
runners verify imports point into the snapshots.

- CPU: `/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage1-20260926-r2`,
  job `39188573`, `sdfmilan272`, `ps_20241122`, `PS_PARALLEL=mpi`.
- GPU: `/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage1-20260926-r3`,
  job `39188729`, `sdfampere001`, A100-SXM4-40GB, CuPy 13.6.0, CUDA runtime
  12090, `PS_PARALLEL=none`, KvikIO compatibility mode ON. This validates
  correctness with fallback I/O, not GDS performance.

| Check | Result |
|---|---|
| Core `pytest psana/psana/tests/` | 484 passed, 114 skipped, 10 deselected; 99.45 s |
| Core `pytest psana/psana/tests/byhand_*` | 4 passed; 115.43 s |
| A100 integration, including all eight slow pixel-exact cases | 117 passed; 682.50 s |

The CPU run uses the same runtime sources as r3; only the GPU admission test
fixture was corrected between snapshots. No runtime edit was needed for that
failure.

New tests cover host-only setup without CuPy/calibration; unsupported layouts;
sparse/reversed segment order; raw gain bits; missing sources; malformed runtime
dimensions; cross-stream consumers; slot reuse; budget rejection; and one gather
per subbatch. Legacy calibration/geometry entry points are poisoned during the
raw-only device test. Existing device tests cover multi-owner gather, allocation,
retirement/failure handling, transitions, and default pixel-exact calibration.

The first GPU attempt, job `39188574` in r2, passed 108 tests and exposed an old
test-fixture mismatch: `test_gpu_admission_device.py` uses per-dgram submission
without an input-group pool, but left the reader's default bulk mode enabled.
The test now explicitly selects bulk-off on reader and manager; production I/O
code is unchanged. That attempt deselected eight slow tests. The r3 run clears
pytest's default `not slow` filter to include actual-data pixel-exact checks.
Snapshot preparation r1 stopped on the login host's Python 3.6 subprocess
argument incompatibility before test submission; its partial snapshot is unused.

## Stage 2 boundary

The raw-only component is exercised internally, not selected by DataSource yet.
Stage 2 must wire task declarations and exact constant uploads, select the
preparer in manager setup, and consistently bypass MPI's earlier derived CPU
calibration/geometry and fixed-pair CUDA IPC for callback-only detectors.
CPU/hybrid consumers and collective ordering must remain intact. Existing MPI
setup routines were not disabled or changed in Stage 1.

The preparer accepts the admitted subbatch; it makes no calibration-based batch
choice. For callback mode with an unspecified batch size, use the existing
simple fallback of one at wiring time, subject to byte admission, rather than
calling `optimal_kernel_batch_size()`. Explicit batch sizes remain supported.
Generic fields retain their descriptor path; additional dense adapters require
their own layout contract.
