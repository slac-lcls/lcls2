# Cancelled submission before allocation

The post-review 105-test run failed in the existing timeout test: it expected a
child-created PID file within a 2-second process timeout, but that file was absent.
The submission shell lacked fail-fast handling and continued after the coordinator
assertion, submitting job 61534921. The coordinator immediately cancelled it.
Final sacct: CANCELLED by 73802, elapsed 00:00:00, no node assigned; direct
scontrol query found no steps. No compute workload ran in this attempt.

The failed log is retained as `offline-final.log` in task scratch. It is not a
passing result. The test now captures the PID from the real Popen object, checks
the timeout still occurs, and verifies the real child is reaped/nonexistent,
removing reliance on child interpreter startup and scratch-file latency. This
changes only the test; production timeout behavior remains unchanged.

Validation, commit and submission are separate checked actions on retry. This
incident is retained in the final evidence; it is not a discarded hardware cohort.
