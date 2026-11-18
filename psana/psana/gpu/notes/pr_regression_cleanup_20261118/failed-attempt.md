Initial implementation df42d1cd6, job 61502056 on nid008244: FAILED/1:0
in 9 seconds before injection. All three target ranks and the sentinel started;
the harness rejected missing live-step identity. No case pass/complete receipt.
The old squeue-based matching helper returned no numeric matches, so its empty
cleanup lists cannot themselves prove absence. Final job accounting shows both
numeric steps CANCELLED and the allocation ended. Dataset was not accessed.

Diagnostic job 61502138 on nid008244: COMPLETED/0:0 in 24 seconds. Two separate
20-second sleep steps COMPLETED/0:0. While running, both narrow and wide squeue
output showed only batch/extern, whereas scontrol show step returned both live
numeric steps with full names. This establishes an observable discrepancy;
we have not established the underlying Slurm cause. Raw snapshots are included.

Fix in 86af716c8: shared matching_steps uses direct scontrol --oneliner show step,
validates each record and exact full name, rejects foreign/duplicate/malformed
identities and empty response. Cancellation remains scoped to matching numeric
step IDs, with the unchanged six-second total deadline. Harness saves raw live
snapshots before injection and after target exit. 31 adapter and 5 harness tests
PASS on this implementation. Prior site placement job still has successful sacct
records; its old squeue absence observations alone are not cleanup qualification.
Retry 61502208 uses fresh frozen-02 sources and fresh output directory.
