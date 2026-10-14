# User-kernel Stage 1b: remove built-in GPU calibration

2026-09-26. Implemented on `codex/psana2-gpu-user-kernels`, relative to
`1d484d43d`. This completes the
[removal audit](proposals/calibration_runtime_removal_20260926.md).
The public callback/publication API remains unimplemented (Stages 2–4).

## Runtime change

GPU setup now resolves parsed fields directly from Configure, without creating
a CPU Detector, looking up pedestal shapes, computing derived constants/masks,
or uploading geometry. GPU input routing works with no calibration dictionary.
The default event path exposes parsed inputs and generates no synthetic `.calib`,
`.raw`, or `.image` outputs. Pre-calibrated float32 XTC fields remain inputs.

Removed `GPUDetector`, its output buffers, calibration/cleanup launches, shape
heuristic, geometry helpers, BeginStep recipe, fixed-pair CUDA IPC and leader
selection, and the float32 image D2H pipeline. `gpu_calib.py` is removed.
A numerical reference and CUDA header are retained only under `tests/gpu`.
They are not imported by the runtime. Generic allocation uploads remain in
`gpu_allocation.py`.

`DenseInputPreparer` remains a budgeted input component. Internal callers can
select it through `EventPool`/manager input preparers; the default manager does
not gather unrequested detector images. Prepared input batches stay attached to
execution records through consumer completion. They are inputs, not published
results. The same batched parser and one-gather-per-prepared-subbatch structure
is preserved. No extra kernels or CPU per-field launch loop were introduced.

MPI still assigns devices and sizes per-BD quotas. GPU-exclusive detector names
are filtered consistently from derived CPU shared calibration/geometry caches
on every participating rank. CPU and hybrid consumers retain their caches,
source dictionaries, and CPU calibration APIs. BeginStep/EndRun still drain
prior input users before host transition dispatch; no GPU calibration refresh
runs. Source calibration loading/distribution is retained for CPU consumers and
future explicit constant declarations.

## Compatibility and examples

- `gpu_d2h_chunk_size` accepts only its retired zero default; nonzero requests
  fail before run setup. There is no automatic output D2H.
- Non-None `gpu_fn` fails explicitly until task support is implemented.
- Use `evt.gpu.detector(name).field(algorithm, field)` for parsed inputs.
  Its `.on_cpu` is an explicit input copy; `.on_gpu_view(stream)` uses the
  existing input ownership/completion contract.
- The [input-only example](../examples/input_only.py) demonstrates the current
  API. The multi-GPU smoke driver now checks parsed input availability.
- Automatic-calibration benchmark entry points are retired, and calibration
  sweep harnesses reject unsupported workloads. Their implementations/results
  at `1d484d43d` remain historical evidence, not input-only performance numbers.
  Feespec-only benchmark variants remain available. Untracked experiments were
  excluded from this change.

## Validation

Frozen snapshots are under
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage1b-20260926-r2`
and `...-r3`. Each includes a source manifest and patch. Python source/package
data were overlaid on the verified master-merge installation; removed sources
were explicitly pruned. No native extension changed or needed rebuilding.
Import checks confirm the snapshot is used. Environment: `ps_20241122`;
CuPy 13.6.0, CUDA runtime 12090 on A100, KvikIO compatibility mode ON.
These are correctness checks with fallback I/O, not GDS or performance evidence.

| Check | Job / host | Result |
|---|---|---|
| Core `pytest psana/psana/tests/` | `39190919`, `sdfmilan271`, r2 | 473 passed, 105 skipped, 7 deselected; 154.81 s |
| Core `pytest psana/psana/tests/byhand_*` | `39190919`, r2 | 4 passed; 119.25 s |
| Full A100 integration, including slow real-data cases | `39190920`, `sdfampere001`, r2 | 105 passed; 216.17 s |
| Four-rank MPI, exclusive and hybrid | `39190921`, `sdfampere011`, r2 | Both passed; 13 events each, both BDs active on one GPU |
| Final result-access tests and example CLI | `39191285`, `sdfmilan270`, r3 | 34 passed; 0.24 s; example `--help` passed |

Real-data cases use public `mfx100848724` run 51, 13 events, across per-dgram and
bulk I/O, one/two slots, and a partial batch tail. GPU-exclusive tests omit
calibration loading and check raw pixels exactly against CPU values. Hybrid
cases also compare normal CPU calibration with the reference. The separate
`mpi_input_only.py` driver compares raw hashes and hybrid CPU calibration hashes
against a serial reference, with all four ranks sharing one CPU shared-memory
group. Its target audit verifies exclusive detectors are absent and hybrid
detectors remain present in both derived-cache loops on all four ranks.

Device tests also exercise missing/corrupt fields, preserved gain bits,
sparse segment IDs, batched launch counts, multiple input owners, delayed
consumers, allocation growth/admission, transition drains, failures, and early
close without calibrated outputs. Deleted tests exclusively exercised the
removed image-D2H implementation; their broader ownership checks remain.

The first snapshot (r1: CPU `39190680`, GPU `39190681`) exposed fixture migration
errors: legacy result keys/tuple adapters, stale presence/completion expectations,
an over-renamed mock method, and abstract test construction. These were corrected
before r2; r1's byhand suite already passed. The final r3 change updates only
result error messages/docs and removes the geometry-specific missing-result
message; the core GPU input and MPI implementation matches r2. The first r3
wrapper (`39191213`) failed to activate the environment under `/bin/sh`; the
replacement explicitly runs Bash.

## Lines removed

Counts are physical lines from `git diff --no-renames --numstat 1d484d43d`, including comments
and blank lines. Runtime means the changed GPU package implementation/CUDA header
and `psexp/ds_base.py`/`mpi_ds.py`; it excludes benchmarks, examples, tests, and docs.

| Runtime lines deleted | Runtime lines added | Net runtime reduction |
|---:|---:|---:|
| 1,877 | 95 | **1,782** |

The 201 lines of numerical reference/header now in test support are outside the
runtime count; this is a runtime removal, not a claim that all algorithm source
was discarded. Retired benchmark code and documentation reductions are excluded
from the reported runtime saving.
