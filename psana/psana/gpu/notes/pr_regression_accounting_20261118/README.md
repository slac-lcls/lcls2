# Run-51 full-prefix cache and traffic accounting

Status: **all six 10k cache/byte-accounting cases passed** in job 61537210 on
nid008505, COMPLETED 0:0 after 51:53. Independent coordinator audit verified 380
indexed capture files and 64 placement files. All 19 numbered steps plus batch,
extern and allocation completed 0:0; a final direct query found no remaining steps.
**Final Claude review: PASS, no blockers** ([review](review-03.json),
[response](response-03.md)). All 109 tests passed on the exact reviewed commit
([validation](validation-06.json)). No A/A, A/B, timing-equivalence or
component-bound pass; all shipped qualification flags remain false.

This follows the six-cell 1k smoke with the same private run 51 copy and pinned
runtime 1995752058bb9aaa3f16ccd86c22e84153257538. It verifies the selected
10,000-event prefix (335571760000 bytes, 312.526 GiB) in each raw case, with separate
200-event warmups. Parameters remain batch 20, one GPU stream per BD, 8 GiB budget,
bulk reads of 1 MiB, KvikIO 4 workers / 1 MiB tasks / 16 MiB bounce, compatibility mode ON,
GDS off, SMD 1000 and OMP 1. GPU cases use one physical GPU and dense gather plus
metadata callback; no user analysis kernel, calibrated-pixel or output/D2H claim.

## Preserved attempts

| Job | Result | Evidence |
| --- | --- | --- |
|61534921|Cancelled while pending, no compute|[Submission incident](submission-incident.md)|
|61534960|First measured CPU BD1 exceeded 300 s; no case accepted|[Attempt 1](attempt-01.md)|
|61535808|Four event cases pass; CPU steps BD1 fails cold-cache setup; BD2 steps unrun|[Attempt 2](attempt-02.md), [audited partial evidence](attempt-02-evidence.json)|
|61537210|All six cases PASS; COMPLETED 0:0|[Full audited evidence](evidence.json), [final validation](final-validation.json), [Slurm accounting](job-61537210-final-accounting.txt)|

Final job 61537210 results below establish raw correctness/accounting for this
pinned runtime and selected prefix. Each case processed 10,000 events and exactly
335571760000 selected bytes, after a separate 200-event warmup. Both BDs performed
work where configured. The last cache attempt must have two zero observations.

| Case | Events per BD | Cache attempts | Final resident bytes | HNI RX / selected bytes |
| --- | --- | ---: | --- | ---: |
|cpu-raw-events-bd1|[10000]|1|[0, 0]|1.2990|
|gpu-raw-events-bd1|[10000]|1|[0, 0]|1.3741|
|cpu-raw-events-bd2|[5010, 4990]|2|[0, 0]|1.3366|
|gpu-raw-events-bd2|[4960, 5040]|1|[0, 0]|1.0678|
|cpu-raw-steps-bd1|[10000]|2|[0, 0]|1.2959|
|cpu-raw-steps-bd2|[4993, 5007]|1|[0, 0]|1.3375|

CPU events BD2 initially retained 16,384 bytes and CPU steps BD1 retained 98,304
bytes, each in both observations. Their second attempts were zero/zero and only
then began measurement. All other cases passed conditioning on attempt 1. The
per-range observations and hashes remain in the indexed capture. These observations
do not establish why eviction initially left those pages resident.

Production source commit: `2158556ebf99ede85ae872343b5687b0f9c1c6f1`, verified
against [frozen checksums](frozen-03-SHA256SUMS) and the source recorded in the
capture. [Batch script](accounting-03.sbatch). Timings in the JSON are instrumented
correctness diagnostics, not clean throughput or regression comparisons.

## What the controls establish

All three allocated attempts passed two 256 MiB cold/warm controls plus idle observation:
HNI RX follows cold reads while warm/idle RX stays low. Kernel netdev counters
miss most of this traffic. HNI is therefore the useful empirical node-traffic
coverage observation here. Whole-node counters still include protocol/background
traffic and read-ahead; they are not datagram equality, storage-only attribution,
physical media reads, or a one-OST/link ceiling. Driver counter timestamps are
read times, not an independent firmware-freshness guarantee.

MPI controls show a pending SMD0 termination receive at function return, the
builtin-world pending-reference warning in the observed case, and completed
Cancel/Wait with no warning in the isolated cancellation case. Observation retains
the request/buffer longer than normal code. This locates a specific cleanup path;
it does not fix installed psana or qualify full MPI lifecycle behavior. Normal
workloads keep the original runtime and its warning.

## Reproduction and remaining work

Use [ACCOUNTING.md](../../scripts/performance/regression/ACCOUNTING.md), frozen
source checksums and the exact batch script for each attempt. Raw captures are
under `/pscratch/sd/m/monarin/psana2-gpu/pr-regression-accounting-20261118`.
The coordinator collector reads receipts/metadata, verifies Git and frozen-source
pins, selected input metadata and accepted hash receipts, case byte/event/cache
checks and traffic coverage. It does not reread large XTC payloads on login nodes.
Failed captures retain separate indexes and checkpoints. Source/input hashes and
scheduler accounting remain with each attempt; failures are never relabeled.

Current protocol allows three cache-conditioning attempts before measurement,
with two zero-residency observations required and every attempt saved. Failure
after three attempts stops the campaign. Full steps are bounded at 900 s,
controllers at 1950 s, allocation at 5 h. Per-rank 120 s stack diagnostics are armed, and
stage receipts distinguish setup/cache/iteration. Timings are instrumented.

Final code/evidence review completed in round 3 of 3. Subsequent work remains
instrumentation-overhead qualification, component
counter/bound qualification, and clean A/A before a selected-PR A/B. The MPI
cleanup finding belongs in a separately scoped production-source change with
existing lifecycle correctness coverage. SDF adapter support is still separate.
