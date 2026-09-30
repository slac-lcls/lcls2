# Stage 6: first-version user-kernel lifecycle acceptance

**Status: accepted.** No production runtime changes or blocking runtime findings in this stage.
Runtime/algorithms are `6ba5fa586`; checkout `5e9e134aa` adds documentation only.
Stage 6 adds eight GPU pytest cases and a four-rank MPI lifecycle driver,
then refreshes the completed performance evidence. Validation totals are
**608 CPU tests, 5 longer MPI tests, 173 GPU tests, and 18 public MPI cases**.
The main suite also reported 173 skips and 7 deselections; the device suite
ran all 173 GPU cases without skips. No additional performance campaign was
needed because production code and the scientific algorithms are unchanged.

## Acceptance matrix

The full device and CPU suites and all 18 public MPI cases passed.
The following tests cover the acceptance contracts.
Test paths are relative to `psana/psana/tests/gpu`.

| Contract | Evidence |
| --- | --- |
| Host-only declaration; selectors fail early; original selective constants; no implicit calibration | `unit/test_gpu_task.py`, `unit/test_core.py`, `integration/test_gpu_task_device.py` |
| One callback per selected execution subbatch; one producer completion; tails, multiple inputs and segment ordering | `integration/test_gpu_producer_device.py` |
| Sparse, empty and changing output names/shapes/dtypes; retained host results | `unit/test_gpu_d2h.py`, `integration/test_gpu_d2h_device.py` |
| Delayed D2H blocks reuse; bounded pinned cache and pageable fallback; failed completion retains owners | `integration/test_gpu_d2h_device.py`, `integration/test_gpu_producer_device.py` |
| Read groups, independent owners, memory pressure, delayed external consumers | `integration/test_bulk_lifecycle_device.py`, `integration/test_multiowner_gather.py`, `integration/test_multiowner_transitions.py`, existing budget/admission unit tests |
| Actual task plus D2H drains before BeginStep refresh; next callback sees new constants/generation; old result remains unchanged | **New:** `integration/test_gpu_task_lifecycle_device.py`, with pinned caps 0/8192 |
| Actual task plus D2H drains before EndRun dispatch | **New:** same module; existing bulk lifecycle test checks once-only EndRun handling |
| Mutating registered owned copies preserves borrowed inputs/constants through overlapping slots, tails, reuse and public field access | **New:** same module, depths 1/2 |
| Valid non-Corrupted damaged input stays present; Corrupted input is absent and zero-filled | **New:** same module, plus batched/lazy parser equivalence in `integration/test_batched_locators.py` |
| Public serial early close, break and loop-body exception retire owners and preserve outputs | `integration/test_gpu_d2h_device.py`, caps 0/8192 |
| Public MPI exclusive/hybrid delivery and max-events tail | `integration/mpi_gpu_publication.py`, four ranks/two BDs sharing one A100 |
| Public MPI explicit close, scoped break and scoped loop-body error; pending outputs retained after close | **New:** `integration/mpi_gpu_lifecycle.py`, exclusive/hybrid × bulk off/on |
| Fatal callback error after launch drains local GPU owners and aborts MPI without hanging | **New:** same driver in callback-failure mode; launcher requires expected fatal marker, cleanup marker, nonzero exit and no timeout |
| External calibration equals CPU v3; combined calibration/integration retains device intermediates | Stage 5a/5b device reference tests rerun in the full suite; public serial/MPI science-driver evidence remains from Stage 5b job 39375921 |
| CPU-only compatibility | Main psana suite and explicit `byhand_*` MPI suite |

Synthetic transition tests invoke the production transition, task, slot and D2H
paths; transition packets and the host constant update are controlled fixtures.
They complement real-data public serial/MPI tests rather than claiming a real
scan acquisition changed calibration constants.

## Validation provenance

Frozen root:
`/sdf/scratch/users/m/monarin/gpu-validation/user-kernel-stage6-20260929-r1`.
`manifest.json` hashes the frozen source; `source.txt` records the base commit
and tracked test changes. New test source is also included in the manifest.
Job scripts, environment, logs and results are in the same root.

- Job **39479128**, `sdfampere018`: **173 device tests passed** (230.33 s),
  then both real-calibration MPI publication modes passed with 13 unique events
  and batches 5/5/3. `GPU_MPI_RESULTS=0` confirms these phases exited successfully.
  The job was deliberately canceled during the following lifecycle matrix;
  its overall scheduler state is **CANCELLED**, not a successful complete job.
- Job **39479129**, `sdfmilan008`: **608 passed, 173 skipped, 7 deselected**
  in the main suite (136.95 s); **5 passed** in `byhand_*` (131.35 s).
  Job completed with exit 0.
- Focused MPI job **39479943**, `sdfampere034`, frozen root ending `-r4`:
  **12 close/break/loop-error cases and 4 expected callback-abort cases passed**
  in 1m58s; job completed with exit 0. Every normal case reached cleanup and
  validated retained outputs; every fatal case produced the exact error and
  cleanup markers and exited 1, without timing out. Two BDs share one A100.
  This driver uses a controlled numeric constant while retaining MPI constant
  distribution and actual device upload. Real calibration remains covered by
  the two completed publication tests. KvikIO uses CPU fallback.
- Python environment `ps_20241122`; native extensions from the retained
  `a-bpp-off-20260923/installs/Integrated` installation.

The [machine-readable evidence](user_kernel_stage6_20260929.json) records job
outcomes, log hashes and test-source hashes. All 1,257 files in each accepted
frozen snapshot were verified after execution; current test files match the
accepted frozen sources.

Two focused harness attempts are excluded from acceptance. Job **39479814**
(`-r2`) failed because the small constant fixture populated only `dsparms`,
which MPI calibration distribution replaced. The fixture now populates the
source `_calib_const` dictionary as well. Job **39479871** (`-r3`) passed its
first cleanup assertions but stalled at teardown because the test observer
retained managers and their Run/shared-window owners on BD ranks only; it was
canceled. The final driver restores the observed method and releases these
private references before Run teardown. The complete `-r4` matrix then passed.
Neither attempt required a production fix.

Review found no blocker for the documented v1 scope. The v1 closeout includes
the Stage 5c benchmark harness and accepted reports, plus the Stage 6 tests
and API documentation. Production runtime and algorithm sources are unchanged.

## First-version scope

See the [public task guide](user_task_results.md) for the API and cleanup example.
`gpu_fn=GpuTask(...)` enables user callbacks. `batch_size` defaults to **1**;
set `batch_size=20` to request batching, subject to byte-budget splitting.
There is no separate scheduling on/off Boolean. `gpu_bulk_read` controls file
read grouping independently. Omitting `gpu_fn` provides input staging without
automatic user calibration or output D2H.

Borrowed device inputs/constants are read-only by contract; arbitrary CUDA
writes are not intercepted. Users register scratch with `keepalive` and outputs
with `publish` before submitting kernels. Output `.on_cpu` owns an independent
NumPy row. The default pinned-output limit is 64 MiB per BD, with ordinary-host
fallback; it does not cap user device scratch or retained NumPy results.

Use `contextlib.closing(run.events())` for early exit. MPI close drains this
rank's work and outstanding EB messages; it is not a collective stop request.
Fatal pipeline callback errors abort MPI. GPU `run.steps()` iteration is outside
v1; step transitions are handled within `run.events()`.

The science example implements Jungfrau CPU-v3 calibration followed by fixed-bin
radial integration in one callback with two kernels. Common-mode correction,
solid-angle/polarization corrections and pixel splitting remain outside this
example. Other detectors and true-GDS acceptance are separate work.

[Stage 5c](user_kernel_stage5c_20260928.md) records accepted matched batching and
JF/JF+feespec scaling results. Stage 6 is correctness acceptance and introduces
no new throughput claim or broad performance campaign.
