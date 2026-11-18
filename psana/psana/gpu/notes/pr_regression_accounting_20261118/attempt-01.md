# Failed first allocated attempt — job 61534960

Job FAILED 1:0 after 13:21 on nid001116. Prefix verification passed (50k
headers; all accepted 10k range hashes), four placement probes passed, traffic
controls passed, and the three MPI controls isolated the pending termination
receive (warnings 0/1/0, pending then successful Cancel/Wait). CPU BD1 warmup
passed. Its measured step .8 exceeded 300 s; the adapter cancelled it and
confirmed no active matching steps. Final direct query found no steps.
No measured case was accepted; later cases did not run. Raw capture and compact
attempt-01-evidence.json preserve the failure and partial diagnostic evidence.

The original duration estimate treated much of the non-loop cost as fixed.
Full-range cache verification and setup can also scale with the prefix, and the
existing final-only receipt did not distinguish cache time from loop time at
failure. We do not infer a performance regression or deadlock from this timeout.

Retry: 600 s step, 1350 s adapter (greater than 2*600+120), same 600 s prefix and
480 s controls, four-hour allocation. Worst bounded stages total about 11,716 s
(3:15:16), leaving about 45 min for setup/evidence overhead. Add stage receipts
and qualification-only 120 s per-rank stack diagnostics to locate delays if they
persist. No package, data, settings or acceptance-bound changes. Fresh allocation
and output required; old failure is never overwritten or relabeled.
