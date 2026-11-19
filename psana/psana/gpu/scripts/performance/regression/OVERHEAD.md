# Instrumentation overhead

`overhead.py` is the qualification experiment for the raw PR runner. It compares
its usual accounting with count-only observation on the **same installed psana**.
It cannot emit accepted PR campaign results or enable qualification flags.

| Observation | Minimal | Full |
| --- | --- | --- |
| Event / GPU callback counts | Yes | Yes |
| Exact timestamps, native steps, CPU datagram bytes | Unavailable | Checked |
| Delivery-window/transition counters, first-batch timing | Unavailable | Checked |
| Reader pre-reset and final-window accounting | Unavailable | Checked |
| Per-callback size list | Unavailable | Checked |
| Boundary timings, provenance, CPU/GPU placement, cache, NIC/CXI | Same | Same |

Both execute native CPU raw materialization or GPU dense gather plus callback,
with no user kernel, calibration or output D2H. Native psana's own instrumentation
remains present. The comparison estimates the **incremental runner observer cost**
relative to event/callback counting and common boundary checks; it does not
measure those common costs or claim a completely uninstrumented baseline.
The minimal/full switch belongs to `native_worker.py`, not the production
DataSource API. Its event/callback recording and temporary `DeliveryAccounting`
and `ReaderAccounting` wrappers are benchmark additions. Normal production
applications do not activate these wrappers automatically. Native reader counters
and timers still run in both modes; full observation additionally snapshots and
aggregates those counters. An unstable full/minimal timing difference must be
reported as an uncertain overhead estimate, not proof that instrumentation
itself caused the variability. See the [original and retry results](../../../notes/pr_regression_overhead_20261118/README.md).
Minimal samples cannot establish exact per-event identity/bytes. Full samples
must pass the existing independent-reference checks. Input hashes, settings,
pinned runtime and total counts remain checked in both modes. Do not reuse a
minimal sample as correctness evidence or a PR throughput result.

Fixed qualification: all six raw cases, 10k measured / 200 separate warmup events,
six adjacent pairs per case, alternating minimal/full and full/minimal. Every
sample gets cache conditioning after setup and before its synchronized timed
loop. Two zero-residency observations are mandatory; preserve all conditioning
attempts. Use the existing source-pinned single-node native launcher, CPU binding,
shared physical GPU, scheduler checks and bounded cleanup. No sample replacement, exclusion, optional stopping, cProfile, or periodic stack
tracing is permitted. A flagged case may receive one separately recorded fixed
six-pair follow-up under the policy below.

Primary: full/minimal loop-time percentage difference, paired Student-t 95%
interval. All six intervals must lie within **[-5%, +5%]** to pass the screen, matching the later PR +/-5% tolerance.
This is the policy adopted on 2026-11-19 after inspecting the original +/-1% run;
original plans, samples and decisions remain immutable. Negative out-of-band results
also require investigation. Setup/process intervals are descriptive and do not
establish their own acceptance. Failure or imprecision is not equivalence.
The all-cases requirement is an intersection-union equivalence decision; there
is no claim of simultaneous 95% coverage for the six reported intervals.

First run the fixed 1k/two-pair pilot (24 samples, each with separate warmup).
It validates mechanics and can **never** qualify overhead. Full execution has
72 samples and requires that pilot's completion hash and unchanged source set.
The prior 10k accounting timings imply roughly 8–10 hours for the full run;
plan allocation headroom explicitly. Only a completed reviewed pilot authorizes
progress to that already-scoped measurement, with no automatic resubmissions.

Within an allocated single-node batch shell and the pinned Python/environment:

```bash
python -B -s overhead.py --reviewed-commit REVIEWED_SHA --pilot --inputs /absolute/inputs.json \
  --inputs-sha256 INPUT_SHA --output /absolute/new-pilot
python -B -s overhead.py --reviewed-commit REVIEWED_SHA --inputs /absolute/inputs.json \
  --inputs-sha256 INPUT_SHA --output /absolute/new-full \
  --pilot-output /absolute/new-pilot --pilot-sha256 PILOT_COMPLETE_SHA
```

Freeze and hash `overhead.SOURCES` together before submission. Verify the pinned
input specification, reviewed Git SHA and clean task tree. Check the personal
and project allocation balances and use the approved Perlmutter launch shape.
Do not run the workload or hash bigdata on a login node. `collect OUTPUT` only
rechecks saved sample files and recomputes statistics; it does not execute psana.
Keep the final index/completion hashes and audit allocation terminal state after
exit. `mechanical_status=pass` and `overhead_qualified=true` are different claims.

This experiment leaves production MPI request cleanup deferred. Existing normal
runtime warnings must be preserved, not suppressed or patched in this run. A
qualified overhead result still does not satisfy component-counter bounds,
campaign qualification receipts, or the subsequent clean A/A prerequisite.

The collector requires sealed completion, the exact full capture index, matching
source/input integrity receipts, no failure marker, and a passing final step
audit before emitting a pass or qualified field. It also checks cache-attempt
numbering, hashes, failed predecessors and final-observation equality.

The coordinator must compare every frozen source to its Git blob at the reviewed
SHA and save that association before submission. The driver records the declared
SHA and source directory; it deliberately does not invoke Git from the compute
job or pretend a supplied SHA alone proves the source association.

## Five-percent screen and bounded retry

`python -B -s overhead.py screen ORIGINAL_OUTPUT` audits the complete sealed
original run, preserves its original margin/verdict, and emits a separate +/-5%
assessment. The entire 95% interval must fit inside the band. Every other outcome
is flagged for follow-up, including an apparent speedup or an interval crossing
both limits. A mean within 5% alone is insufficient. Pilots cannot select retries.

One fresh six-pair batch is permitted for **all flagged cases**, retaining the
same 10k events, separate 200-event warmup, alternating order, settings, private
inputs, native runtime, worker, launcher and cache/NIC checks. It may run on a new
node; placement is captured and fixed within that batch. Only the overhead
controller can differ from the original source set, and it requires review and
Git-blob association before launch. The original completed full measurement
supplies the mechanics prerequisite, so no additional pilot is needed when all
workload sources and inputs match. No retry-of-retry is supported.

```bash
python -B -s overhead.py --reviewed-commit REVIEWED_SHA \
  --inputs /absolute/inputs.json --inputs-sha256 INPUT_SHA \
  --retry-parent /absolute/original-full --retry-parent-sha256 ORIGINAL_COMPLETE_SHA \
  --output /absolute/new-retry
```

The retry selects cases from the audited original screen, never from individual
slow samples. It preserves all original and new measurements in separate sealed
captures; it does not pool or replace them, stop early, or repeat until passing.
`retry_cases_passed` describes only the follow-up batch. A retry cannot emit
`overhead_qualified=true` or enable a PR qualification flag. Report the original
and follow-up intervals side by side and review the combined screen disposition
explicitly. Outcome-selected retries do not preserve a single preplanned 95%
coverage claim across attempts. If the follow-up remains outside the band,
report unresolved variability/overhead and investigate before more measurement.
