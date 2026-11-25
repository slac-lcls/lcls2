# Component performance contracts for PR campaigns

The campaign asks whether a PR keeps the production path within acceptable
performance and resource boundaries. It does not recreate the unit/device test
matrix. Keep the current six CPU/GPU raw cases, one GPU stream per BD, with
fixed dataset, settings and topology. Add a workload only when a changed component
is not exercised by those cases (for example calibrated work for constants sharing).
A known multi-stream issue is not a reason to add a multi-stream acceptance case.

This document refines the [agent workflow](pr-regression.md). It defines required
campaign review evidence. The current controller automatically gates loop-time
comparisons and receipt integrity; it does **not** yet collect or automatically
accept/reject component counters. Native adapters and qualified bounds are still
required. A controller-level equivalent label alone is not full PR performance
acceptance under this policy.

## Use the existing tests for correctness

| Responsibility | Existing coverage to reuse | Campaign responsibility |
| --- | --- | --- |
| Batch submission and retirement invariants | [input-group tests](../../../../tests/gpu/unit/test_gpu_input_group.py): test_batch_submission_uses_plan_without_repolling_and_preserves_reads; test_retirement_candidates_skip_live_windows_and_retry_delayed_cuda | Observe native call/visit counts and host cost on the frozen real workload |
| Callback granularity | [producer tests](../../../../tests/gpu/unit/test_gpu_producer.py): test_one_batch_invocation_identity_stream_and_context_expiry | Record actual execution sizes and callbacks in the installed runtime |
| Budget splitting and pressure handling | [admission tests](../../../../tests/gpu/unit/test_gpu_admission.py) and [group scheduling](../../../../tests/gpu/unit/test_gpu_group_schedule.py) | Detect unexpected splitting, pressure drains and higher memory/CPU cost at unchanged inputs and budget |
| Ownership, errors and close/drain behavior | [input-window tests](../../../../tests/gpu/unit/test_gpu_input_window.py), [result lifetime](../../../../tests/gpu/unit/test_gpu_result_lifetime.py), [bulk reads](../../../../tests/gpu/unit/test_gpu_bulk_read.py) | Measure completion/teardown cost and retained-resource behavior during representative execution |
| Device execution and numerical acceptance | [GPU test guide](../../../../tests/gpu/README.md) and its unit/device/slow acceptance suites | Retain exact-revision test receipts; use only enough campaign preflight to prove the measured workload actually ran |

Do not add separate performance cases merely to repeat partial-tail, multi-step,
EndRun, early-close, failure injection, mask or pixel-exact test assertions.
Those stay with the relevant existing unit/device tests. A deterministic new
algorithmic invariant belongs in that suite when missing; it is not an excuse
to build a second correctness implementation inside a benchmark.

The controller's `correctness` phase remains a workload preflight: verify the
installed revision, event identity/count, selected inputs, valid output witness
when applicable, placement and complete execution. Reuse maintained references
and applicable test receipts. A truncated performance prefix is not a substitute
for full transition/lifetime acceptance. A skipped device test is not a pass.

## Preserve the work, measure the scheduling cost

The [input-group study](../input_group_bookkeeping.md) illustrates the target:
50,000 physical reads and 100 callbacks were unchanged, but pool polls fell from
50,101 to 101. Two-stream polling self time fell from 9.22 s to 0.18 s in separate
instrumented runs. Individual candidate visits were **not** counted. These are
historical Sim10 measurements at batch 100, not limits for run51/batch 20, and
not a reason to include two streams in the current screen.

Public event iteration legitimately scales with event count. Reading many file
ranges also legitimately requires many read submissions. Protect batching of
planning, admission, dispatch and bookkeeping; do not require all calls to be
constant or forbid a necessary linear pass over items. In particular, detect
repeated scans of a growing retained/pending set and hidden per-item scheduling
where one admitted-plan submission should suffice.

| Component | Observations | Useful normalization / interpretation |
| --- | --- | --- |
| CPU SMD/EB/BD scheduling | Events, EB packets/batches, actual GPU executions, batch-size distribution, host dispatch/planning calls and host CPU seconds | Calls per admitted plan/execution; CPU seconds per event and batch; retain per-role/rank values |
| Input plan submission | Plan admissions, issue_groups calls, groups submitted, bookkeeping self time | Calls per admitted plan and host cost per group; physical reads can still scale with file ranges |
| Retirement and occupancy | Pool polls, candidate/live-window visits, occupancy/retained-list scans and their work units | Polls per plan/execution; visits per retired window; compare scaling with groups and live-set size |
| Reader | Successful read requests, requested/useful/completed bytes, wait calls and wait time | Requests per input group, bytes per request, read amplification; wait time is not end-to-end I/O throughput |
| GPU task dispatch | Callbacks, execution subbatches and size distribution, kernel launches when available | Callbacks per execution and events per execution; a no-op callback does not test useful kernel/output performance |
| Synchronization and pressure | Host/device synchronization calls, pressure drains, admission retries, allocations/reallocations | Counts per execution and bytes processed; expected nonzero values must follow the actual workload and budget |
| Resource use and sharing | Per-rank CPU time, GPU/pinned/host memory peaks, per-rank work distribution, physical GPU totals | Find extra host work, growing retention or rank imbalance hidden by aggregate throughput; do not double-count shared constants |
| Lifecycle | Explicit setup, first completed batch, drain/close and total process timing | Keep startup and teardown regressions visible separately from sustained throughput |

Use [reader io_stats](../../../gpu_kvikio_read.py) for the counters it actually
exports, and [input-group source](../../../gpu_input_group.py) to select valid
instrumentation hooks for each revision. That source does not provide all the
observations listed above as a public statistics API. Capture absent counters
through reviewed diagnostic instrumentation, or record unavailable. Never fill
missing measurements with zero. Do not copy historical monkeypatches without
checking the current call path and nested-call accounting.

## Predeclare the boundaries

Before candidate execution, freeze a component contract alongside the campaign
profile as a hashed artifact. For each relevant metric, record:

- Component, metric, units, semantics and instrumentation source/version.
- Phase/role/rank coverage, reset/drain boundaries, numerator and denominator.
  Denominators distinguish events, EB packets, admitted plans, physical read
  groups and GPU executions; they are not interchangeable. Retain raw counts.
- Qualified baseline observation and the chosen bound: an absolute count, a
  normalized upper bound, a byte ceiling, or a paired timing tolerance. Record
  why it follows from the workload/algorithm or measured baseline variability.
- Whether a breach blocks performance acceptance or triggers investigation;
  required observations may not silently become optional when unavailable.
- Any declared intentional algorithm/work change, reviewed prospectively.
  Do not change bounds after seeing candidate results to manufacture a pass.

Retain the existing +/-5% paired loop-time equivalence screen. It is not a
universal 5% allowance for every counter. Counts expected to be deterministic
can have structural limits; variable polling/retry counts need workload-specific
bounds. A change from roughly one poll per plan to one per physical item can
require investigation even when storage noise hides the elapsed-time effect.
Conversely, a lower count is not evidence of improvement if work was skipped.
Verify event identities, byte accounting and required outputs before comparing.

For a declared component boundary, report pass/breach/unavailable separately
from the clean timing equivalent/regression/improvement/inconclusive label.
Do not approve a known component-bound breach solely because loop time is
within 5%. If a required bound cannot yet be qualified, leave that part of the
campaign unqualified and state it explicitly; do not invent numerical limits.
Startup/close timing bounds likewise need qualified noise/absolute tolerances;
the current controller reports some secondary timings but does not gate them.

## Two complementary measurements, one workload matrix

1. **Clean A/A then paired A/B:** same six cases and frozen controls; no per-item
   tracing or intrusive monkeypatch timers. Persist only validated low-overhead
   counters, with overhead demonstrated during qualification.
2. **Separate diagnostic evidence:** run the same relevant cases with counters
   and host/component timing. Use the existing `profile` phase for diagnostic
   screening; it runs only one sample per revision/case and cannot establish
   a narrow timing confidence interval. If a component timing gate needs paired
   repetitions, define a separately reviewed repeated diagnostic cohort. Never
   mix these instrumented times into the clean sample pool.

Keep per-rank measurements and whole-case totals; do not compare only averages
that hide one overloaded BD. Sum additive counts/CPU work with explicit scope;
do not add overlapping waits, GPU intervals or independently maximized phases.
When read counters reset per batch, accumulate non-overlapping deltas and retain
the final tail/drain snapshot exactly once. Node NIC totals are contextual,
not proof of attribution to an individual rank or of GPU context-switch cost.

A diagnostic counter jump identifies where to look. A small batch-size scaling
probe can distinguish per-plan from per-item growth when attribution remains
unclear, but it is a targeted follow-up, not a permanent default sweep. Full-node,
calibration, output/D2H and multi-stream extensions require a concrete uncovered
performance mechanism; correctness coverage alone does not justify adding them.

The final PR report includes the clean timing verdict, component-bound table,
per-rank counter ratios, resource observations, explanations for intentional
changes, existing correctness-test receipts and unresolved limitations. Independent
results review considers all of them before declaring performance acceptance.
