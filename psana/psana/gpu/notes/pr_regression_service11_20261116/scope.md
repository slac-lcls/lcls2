# Service-11 observer compatibility

Use the runtime TransitionId.isEvent classifier for the benchmark observer;
bind it once outside the per-envelope loop. Preserve event/window/drain and
transition semantics, and offline operation of adapter tests. Close NAT-009.
One production predicate and its setup wiring; no framework algorithm changes.

Acceptance: adapter/controller tests; real classifier exercised on mixed 11/12
windows, service-11-only first window, transition-only window and final drain;
all three observer hooks restored; descriptive local timing comparing old/new
observer (no timing assertion). Independent Claude review, at most three rounds.
Microbenchmark is not native instrumentation qualification. No jobs, dataset
changes, qualification flags, or frozen campaign changes are authorized here.
Base: 980a7e7e3611c7561e4647af7d2fa0e126da8b94.
