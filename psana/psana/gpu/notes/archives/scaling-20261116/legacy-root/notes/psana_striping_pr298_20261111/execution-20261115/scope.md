# Fixed-profile qualification and sizing

User authorized proceeding after the reviewed configuration selection. Adapt the
historical harness in this separate directory, review it, and run one bounded
qualification allocation: 31 exact-PR GPU integration tests, full metadata/range
validation, six 100-event stripe binding cases, one 1003-event value/tail witness,
and one 100k count2 sizing case. All cases use four BDs, one GPU, one stream/BD,
four workers/BD and batch200. No full stripe screen submission in this phase.

Keep PR298 67c327a unchanged, existing verified payloads and views, strict
input/runtime/handle/cache/event/range/drain checks. New disjoint34-core mapping
must be qualified against actual rank masks. Add sampled whole-device high-water
telemetry and exact framework-budget high-water observations, distinguish those
metrics, and retain at least4GiB observed free device memory. Track each6GiB BD
budget and24GiB aggregate reservation; other CUDA overhead is separate.

Qualification walltime45min, lcls_g/premium. Existing3042node-seconds count toward
the21600second phase cap. Recheck Iris and allocation ledger before submission.
Generate a bounded future screen walltime from the accepted full-case duration;
no automatic promotion or full-screen submission. Preserve failures and charge
their allocation time. No source/build mutation, payload copies or cube work.

Use implement-review-loop, maximum3rounds for this phase, including native
results review if possible. The user explicitly authorized external Claude
review of this task's report/plan; existing task review-loop authorization
covers scoped code/results review. New runtime changes need review before jobs.
Acceptance: meaningful negative audit/affinity/memory/launcher checks, matching
native test inventory, all8cases passing independent audit and Slurm0:0,
memory and callback/tail evidence, and final independent review with no blocking
findings. No performance ranking from qualification cases.
