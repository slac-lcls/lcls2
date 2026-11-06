# Step 3: production callback integration

Continue on tests/smd-callback-chunk-state from 1aff72ca8. Codex implements;
Claude independently reviews exact commits in detached checkouts (three rounds,
15-minute invocation limit). The user approved external Claude code/context
transfer and pushing this branch; the review loop stays active this session.

Connect the persistent callback to serial and MPI production paths, preserving
one callback invocation per run per EB. Bound retained callback output including
reject-all callbacks and transitions. Preserve packet layout, destinations,
step/EPICS propagation, and early-return shutdown. Remove the 16 expected-failure
marks only when all production regression cases pass. Do not change numerical
Jungfrau processing or promise global counters across multiple EB ranks.

Use same-thread cooperative stack suspension via greenlet (new declared runtime
dependency), avoiding callback/MPI migration to a worker thread. Serialize
transition records as they accumulate to release parsed Dgram trees. Bound
output by records, selected-event count and bytes, allowing one oversized record.
On normal callback return, stop selection and drain remaining transitions/input
to complete the upstream protocol. Explicit close must release the callback.

Validate the exact final code SHA with the full regression matrix, meaningful
reject-all/early-return/routing/metadata/backpressure/lifecycle checks, both
required psana pytest groups, and independent review. Carry forward step-2
findings CBI-R1-001 (exact-SHA validation), CBI-R1-002 (cleanup failure priority),
and CBI-R1-003 (parsed-record retention cost).
