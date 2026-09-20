# Implementation/validation status

Historical snapshot at completion of job 38561649 on 2026-09-18. References to
"current source" and "uncommitted" below describe that measurement, before the
memory-reporting and on-demand-wrapper follow-ups. See
`psana/psana/gpu/docs/batched_pre_bulk_review.md` for final review status.

Production implementation is frozen for measurement. Base B+ commit:
`b0c9c3c02`; no new commits or pushes. Frozen installed B+ is in ignored
`install_baseline`, copied before `--python-only` refreshed the current install.
All five relevant modules match baseline Git / current source respectively.

Completed:
- 167 CPU GPU-framework unit tests.
- Job 38559897, sdfampere024: 203 focused CPU/GPU tests, six slow deselected.
- Main suite: 234 passed, 41 skipped, eight deselected (core.log).
- Four longer MPI tests passed (byhand.log).
- Job 38560120, sdfampere014: added delayed-consumer test passed; all six slow
  real-data pixel-exact tests passed (tails, D2H, hybrid routing).
- Seven harness AST/timing tests passed.
- git diff --check; original reference gather and cleanup ASTs unchanged.

Earlier job 38559988 failed only because the new delay test omitted RawKernel's
empty argument tuple. Job 38560054 failed only because the test used query()
instead of the installed CuPy Event.done property. Both test-only mistakes are
fixed; no production edits followed the initial successful correctness run.
An initial byhand invocation with -m slow deselected all cases; rerun without
that incorrect filter passed all four.

Completed primary comparison: **38561649**, sdfampere004, COMPLETED / 0:0,
48:04 allocation. Four alternating clean pairs, one host-timed pair, two
separate Nsight captures, and four CPU-reference preflights. Both audit.py
and summarize.py pass. Every before/after cache check is 100%; installation
hashes remained unchanged. All 19 tracked Python modules in the measured G
installation match current source; the frozen O source matches B+.

- Clean medians: B+ 37.274593 s, gather 28.578059 s; 23.3% lower elapsed time.
- Gather kernels per 20-event execution: 640 → 1; all kernels: 683 → 44.
- Stream waits over 10,000 events: 320,500 → 501.
- Additional H2D traffic: 501 copies / 401,024 bytes; reader tasks unchanged.
- Sampled non-profiler device-memory peaks: 3,633 MiB for both variants.
- Both Nsight exports carry collection-completeness warnings; expected kernel,
  driver-launch and copy counts reconcile. See the report for limitations.
- Final diff check passes. New implementation remains uncommitted for review.

Report: psana/psana/gpu/docs/performance/batched_canonical_gather_sdf.md.
Primary artifacts: job-38561649-warm-ab/{results,audit,summary}.json and traces.

Review/call path: psana/psana/gpu/docs/batched_canonical_gather_review.md.
Bulk read remains deferred: parser location belongs to each input window;
gathering belongs to each detector execution subbatch. This implementation
explicitly supports one parsed owner per execution and requires eager handles.


Update: job 38559942 aborted during the final frozen-B+ control with
`cudaErrorContained: Invalid access of peer GPU memory over nvlink or a hardware error`.
Thirteen samples completed (three full clean pairs, four G clean samples, all
six host-timed samples), but the fourth O control and both traces did not.
Do not call this job complete or use an unmatched four-G/three-O comparison.
The measured production code remained unchanged. The completed retry excluded
sdfampere024, with four clean repetitions, one host-timed pair and two separate
traces (12 measurements), plus four correctness preflights.
The old three-pair host medians are supplemental, not combined with the retry.

Retry job: 38561649, sdfampere004. All installed hashes in the failed job
were rechecked unchanged after failure; failure.json records its partial status.

Compute Sanitizer job 38561752: all 17 gather tests passed, zero errors.
No production changes were needed.
