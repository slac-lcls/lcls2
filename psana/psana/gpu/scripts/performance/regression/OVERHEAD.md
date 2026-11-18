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
Minimal samples cannot establish exact per-event identity/bytes. Full samples
must pass the existing independent-reference checks. Input hashes, settings,
pinned runtime and total counts remain checked in both modes. Do not reuse a
minimal sample as correctness evidence or a PR throughput result.

Fixed qualification: all six raw cases, 10k measured / 200 separate warmup events,
six adjacent pairs per case, alternating minimal/full and full/minimal. Every
sample gets cache conditioning after setup and before its synchronized timed
loop. Two zero-residency observations are mandatory; preserve all conditioning
attempts. Use the existing source-pinned single-node native launcher, CPU binding,
shared physical GPU, scheduler checks and bounded cleanup. No measured retry,
exclusion, optional stopping, cProfile, or periodic stack tracing is permitted.

Primary: full/minimal loop-time percentage difference, paired Student-t 95%
interval. All six intervals must lie within **[-1%, +1%]** to qualify; this is
stricter than the later PR +/-5% equivalence rule. Negative out-of-band results
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
python -B -s overhead.py --pilot --inputs /absolute/inputs.json \
  --inputs-sha256 INPUT_SHA --output /absolute/new-pilot
python -B -s overhead.py --inputs /absolute/inputs.json \
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
