# User-kernel Stage 2: declarations and requested constants

2026-09-27. Implemented on `codex/psana2-gpu-user-kernels`, following the
Stage 1b performance acceptance checkpoint `600669d15`.

## Scope

`psana.gpu.GpuTask(function, inputs=(), calibconst=())` is a frozen, host-only
configuration. `DataSource(gpu_fn=task)` carries it through serial and MPI run
setup. Selectors are validated and deduplicated in declaration order. Requested
detectors must be routed with `gpu_det` or `hybrid_det`. An omitted task batch
size defaults to one; existing no-task defaults and positional `DsParms`
arguments retain their meanings. File, shared-memory, and DRP source paths and
bare callables fail explicitly.

A dense `"jungfrau.raw"` selector configures the existing batched raw/presence
preparer from Configure metadata. A `(detector, algorithm, field)` selector
validates descriptor access without adding a dense gather. Canonical segment
IDs remain available from bindings and dense preparers; constants retain their
original array indexing, including sparse physical segment axes.

Each `(detector, key)` request stages that exact numeric NumPy value from the
run's calibration dictionary on its assigned BD. Metadata stays on the host.
Empty requests upload nothing; gain-only requests do not require pedestals.
There is no casting, plane flattening, gain inversion, mask preparation, segment
reordering, or CUDA IPC sharing. Scalar, empty, strided, and ordinary arrays
retain their shape, dtype, and logical element values. Strided values use a
contiguous host snapshot. Object/text arrays, nonnative endian values, and
unsupported numeric types fail before upload.

Requested constants use allocation-backed per-BD budget accounting. Uploads
finish on their submitting stream before publication. BeginStep drains existing
execution/input consumers, applies the host transition, then compares the source
against the staged snapshot. Changed values are uploaded atomically; unchanged
values do not allocate or synchronize again. Comparison preserves signed-zero
and NaN payload changes. There is no implicit calibration database reload.
Replacement needs room for both generations; a budget or upload failure leaves
the previous published generation intact. Retained aliases keep their charges.
Close releases constants only after the existing consumers drain.

## Review fixes

- Preserve zero-dimensional `()` shapes in the shared upload helper; NumPy's
  `ascontiguousarray` otherwise promotes scalars to `(1,)`.
- Append `gpu_fn` to `DsParms`, preserving earlier positional fields.
- Stage noncontiguous source arrays without changing shape or logical order.
- Compare original bytes when detecting changed constants, including signed
  zero and distinct NaN payloads.
- Reject event processing with a declared task rather than silently ignoring it.

## Validation

The frozen snapshot is
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage2-20260927-r2`.
It contains a source patch, SHA-256 manifest, batch scripts, environment script,
and full logs. All runtime/test sources were checked against the manifest after
validation. Native binaries came from the verified Stage 1b installation; no
native source changed in Stage 2.

| Check | Job / host | Result |
|---|---|---|
| `pytest psana/psana/tests/` | 39269669 / sdfmilan105 | 499 passed, 114 skipped, 7 deselected |
| `pytest psana/psana/tests/byhand_*` | 39269669 / sdfmilan105 | 4 passed |
| Full GPU integration, including slow cases | 39269670 / sdfampere026 | 114 passed |
| Four-rank exclusive and hybrid input acceptance | 39269671 / sdfampere023 | Both passed; 13 unique events across two BDs, raw/CPU-calibration digests match the reference |
| Four-rank task setup | 39269671 / sdfampere023 | Passed; SMD0/EB do not import CuPy, both BDs stage only requested gain values and dense input metadata |

All three final jobs completed with exit code zero.

GPU validation used an A100-SXM4-40GB, CuPy 13.6.0, CUDA runtime 12.9,
Python 3.9, and public `mfx100848724` run 51. KvikIO compatibility mode was
explicitly enabled, so this is device/lifetime correctness evidence, not GDS
performance evidence. Unit checks cover declarations, unsupported values/routes,
empty requests, sparse segment identity, exact refresh, and transactional
failure. Device checks cover scalar/empty/strided/complex/bool values, retained
aliases, budget failures, real serial setup, and delayed transition consumers.

The first GPU/MPI attempt on sdfampere003 (39269389/39269390) was canceled after
a standalone `cupy.empty(1)` reproduced `cudaErrorDevicesUnavailable`. The retry
excluded that node and added a real allocation preflight. The initial CPU run
39269388 passed too, but final acceptance uses the corrected r2 sources above.
CPU runs report the existing pytest-asyncio configuration warning; GPU runs
report KvikIO fallback warnings. MPI input drivers also emit UCX unmatched-tag
shutdown warnings after their success markers and return successfully.

## Remaining work

Stage 2 does **not** invoke callbacks or deliver published outputs. Attempting
task event processing raises `NotImplementedError` and the serial manager closes
its staged resources. Stages 3 and 4 implement callback dispatch, owner retention,
generic publication, and host delivery. No-task parsed-input processing and
normal CPU/hybrid calibration remain supported. These checks establish
correctness; they are not a new throughput comparison.
