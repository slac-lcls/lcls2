# Component performance policy review, round 1 of 3

Base ce1b9cdba323574efb76bf7dac2d7660e9e30ef5.
Reviewed commit 4d59c55e3501fdd68371aec83d88c389ee58b1e0.
Read scope/status/validation in psana/psana/gpu/notes/pr_regression_component_policy_20261116/.
Changed authored files:
- psana/psana/gpu/docs/performance/templates/component-regression.md (new)
- psana/psana/gpu/docs/performance/templates/pr-regression.md
- psana/psana/gpu/scripts/performance/regression/README.md
Plus task bookkeeping, no code/profile/runtime/test changes.

User direction: avoid duplicating unit tests in the PR performance template;
protect main components within acceptable performance boundaries, especially
regressions from per-event/per-item host scheduling where batch scheduling is
possible. Counters previously exposed this clearly. Prior steering retains CPU
1/2-BD and GPU1/2-BD-on-one-GPU cases, omits multi-stream concurrency. This update
keeps the existing six-case matrix and one stream per BD.

Scope: agent guidance and component-performance contract only. Existing generic
controller remains unchanged (loop +/-5% paired equivalence and receipt checks);
it does not collect/enforce component counters. Native adapters remain unimplemented.
The docs must explicitly distinguish required future adapter evidence/manual
results review from currently automated gates. Do not claim this update implements
new counters, metrics, timing gates, hardware results or an end-to-end runner.
No new numerical bounds are invented for counters. Same frozen workload across
clean and diagnostic cohorts, instrumented timing excluded from clean pool.

Inspect cited tests and runtime source to verify boundary claims, especially
unit/test_gpu_input_group.py batch submission without repolling and retirement
candidate checks; test_gpu_producer.py one callback per execution; budget,
ownership and integration acceptance links. Source under psana/psana/tests/gpu/.
Historical input_group_bookkeeping.md records 50101->101 polls with unchanged
50000 physical reads and100callbacks; polling self time9.22->0.18s is instrumented
two-stream evidence, candidate visits unmeasured, not transferable limits for
run51/batch20. The new docs say all of this and retain one stream in default screen.

Review semantic denominators (events vs EB batches vs plans vs groups vs GPU
executions), legitimate per-event iteration and per-range I/O, native counts vs
unit invariants, per-rank work vs misleading aggregates, CPU self-time vs overlapping
waits, cache/control independence, explicit unavailable observations, prospectively
qualified bounds, and limits of profile's one sample per side/case. Changed
components not exercised by raw staging may need a separate workload; no generic
expansion into a second correctness suite.

Validation:18 local links resolve;3 named test functions exist; controller/profile/
tests unchanged by git diff; catalog passes10campaigns/2082hashes/275arithmetic;
whitespace check passes. Source inspection only; no unit/device/native tests rerun
for documentation. Read-only reviewer, no shell/delegation/nested review loop.
Return exactSHA, verdict, stable finding IDs, blocking flags and limitations.
