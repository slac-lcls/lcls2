Final (round3 of3) independent review at f5f45d34adfd8ce58ca34145764e990b6ef88511.
Previous review1af57a40073306763f013dd0876b57554fc53b2d; originalbase5a76c182f.
Same implementation-only scope and guidance as prior contexts. Do not delegate,
execute, edit or start a loop. Passing code review is NOT hardware qualification.

Read psana/psana/gpu/notes/pr_regression_native_adapter_20261116/scope.md,
context-01.md,context-02.md,review-02.json,response-02.md,validation-03.json.
NAT-001..007 were resolved in round2. NAT-008 was the remaining blocker.
Please verify NAT-008 and inspect the correction for new regressions.

Correction: DeliveryAccounting now observes each native Events.__next__ return
and uses psana.utils.first_service to classify L1(service12) versus transitions.
Separate current/window/drain counters, transition counts by service. Classified
envelope totals must agree with the internal _batch_event_count, but ONLY L1
populations must agree with delivered user timestamps and set first-batch time.
GPU final drain snapshots remain idempotent. Restores all three hooked methods.
No envelope content or source ordering is modified. The added per-envelope
observation cost is EXPLICITLY unqualified; docs make no negligible-overhead claim.

Tests:25 adapter tests pass, including original GPU deferred/dain sequence with
classification and method restoration; new CPU mixed BeginStep/Enable/L1/EndStep/
EndRun windows consumed as events and steps. Transition-only first window does
not set first-batch time; L1 population mismatch still fails.20 existing controller
tests pass. Catalog10/2082/275 pass; diffcheckpass; exact Git diff from originalbase
confirms default profile and frozen PR187/208,PR299,notes/archives unchanged.
Supplied test evidence is coordinator-generated, not an invitation to execute.
Source runtime APIs included in this checkout remain the authority.

Only native_worker.py,test_native.py,NATIVE.md and the assemble timing description
changed in this correction (plus review evidence). No jobs, inputs, runtime builds,
qualification flags, Slurm control changes, or performance claims. Runtime/input
manifests, native/API smoke, exact Slurm launch viability and real cleanup tests,
instrumentation overhead, missing GPU/pinned peaks/visit counters and component
bounds remain future qualification. Distinguish these honest limitations from
concrete code defects. Report denied reads and retain stable finding IDs.
