# Stage 4 follow-up: controller lifecycle and transition drains

2026-09-25, branch `codex/psana2-gpu-bulk-batched-integration`, based on
`c8f6b6cdf` plus the existing working-tree bulk-target and benchmark changes.

## Fixed transition ordering

The new controller regression reproduced a gap in `_handle_steps`:
`EventPool.flush()` retires execution leases, but group input leases transfer
terminal events to deferred `InputWindow` owners. A closed input-view context
could therefore still have CUDA work pending when BeginStep/EndRun dispatched.

After flushing executions at those transitions, the manager now calls
`InputGroupPool.drain_idle()`. Unreferenced windows finish their producer and
consumer events before transition dispatch. A drain failure leaves the owner
reachable and charged, prevents dispatch, and allows retry. Live or planned
uses remain protected; this does not cancel a caller's still-open context.
Ordinary event delivery continues to use nonblocking group collection.

## Validation

- **400 CPU unit tests passed**, including 12 additional production-controller
  cases: partial submit/get/short-read failures, parser failure, three
  `max_events` boundaries, missing streams with partial tails across two EB
  batches, minimum working-set rejection before I/O, retained-child pressure
  and close retry, transition ordering, and failed transition-drain retry.
- **10 A100 tests passed**, job **39068235**, node **sdfampere012**, Slurm
  COMPLETED, exit `0:0`, elapsed 21 seconds. This includes
  two new real-CUDA BeginStep/EndRun regressions with a delayed input consumer,
  the existing group pixel/batched-launch checks, partial parser setup and
  drain-failure cases, independent out-of-order reclamation, and multi-owner
  calibration transition checks. Frozen sources were hashed before and after.
- Benchmark harness: **13 tests passed**, including cProfile call-graph parsing
  and separation of instrumented timings from control throughput.
- Python syntax checks and `git diff --check` passed.

CPU validation used `setup_env.sh`, the frozen size4 Python/native installation,
the current source GPU module path, and `PS_PARALLEL=mpi` with one process.
The initial broad CPU invocation with `PS_PARALLEL=none` failed an existing
MPI-specific test because `psana.psexp.node.MPI` was absent; the correct MPI
environment passed the entire suite. No live calibration service was needed.

The device fixture uses actual KvikIO CPU fallback, GPU parsing, input leases,
and CUDA completion. New transition tests isolate dispatch ordering; existing
multi-owner tests also validate changed calibration constants and pixels.

## Artifacts and scope

Frozen device overlay, tests, fixture/source hashes, runner, and job log:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-stage4-correctness-20260925`.

The separate profiling campaign uses the pre-fix frozen size4 runtime so its
paired samples have identical provenance:
`/sdf/scratch/users/m/monarin/gpu-validation/stream-read-profile-20260925`.

This closes a specific transition gap and expands controller coverage. It does
not complete all Stage 4 acceptance: broader real-GPU retained-view/tight-budget
matrices and selective field-owner leases remain to be evaluated. Longer 10k
acceptance and legacy branch/test cleanup remain separately gated in
[the cleanup checklist](stream_read_refactor_cleanup.md).
