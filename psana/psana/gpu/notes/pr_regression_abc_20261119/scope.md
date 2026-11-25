# Three-mode overhead measurement: GPU events, one BD

User approved 12 rounds of A/B/C for the unresolved GPU events, one-BD case and
requested submission. This is a new observer design, not another selected retry
of the two-mode experiment. Use the maintained regression runner, immutable
private mfx100848724 run-51 data and pinned installed runtime. No production
psana changes, MPI cleanup fix, PR A/B, extra topology or qualification promotion.

A: boundary-only benchmark observation; native production counters remain, but
no benchmark event/callback counting or detail hooks. Its observed counts,
identities and bytes must be null, not copied from the request. B: event/callback
counting only. C: full runner detail. All declare the same GPU raw input/gather
and run without an application kernel, calibration or output D2H. Same native
runtime, dataset, settings, node/GPU/CPU binding, warmup and cache/NIC machinery.

Measure synchronized loop wall time and native process wall time in every mode.
The loop includes drain and end barrier. Setup and native process time are
secondary/descriptive. Native process time spans launch to exit and includes
setup, provenance checks, cache conditioning and cleanup, but excludes the
separate warmup process, prefix/site preparation and scheduler confirmation.
Common boundary checks/native timers remain unmeasured costs in A. Do not claim
comparison to an entirely uninstrumented production application.

Fixed main order: ABC BCA CAB CBA ACB BAC BAC ACB CBA CAB BCA ABC. Twelve triples,
36 measured samples, 10,000 events requested/sample, 200-event separate-process
warmup before each sample, private-only cache conditioning with two verified
zero-residency observations, NIC coverage and source/runtime provenance checks.
Compare B/A, C/A and C/B within each triple; paired percent-change Student-t 95%
intervals. Loop screen passes only if all three intervals fit within +/-5%.
Intervals are individual, not a simultaneous 95% coverage claim. No exclusion,
replacement, pooling with old data, early stopping or automatic rerun.

As a prerequisite inside the same submitted allocation, run a six-sample
1,000-event preflight (ABC, BCA) with separate 200-event warmups. It validates
mechanics, not precision, and never passes the screen. Main execution requires
its pinned sealed completion and matching sources/inputs/design. Abort on any
preflight failure. A's workload equivalence rests on unchanged declared input,
iteration/drain, pinned data/runtime and NIC coverage; it cannot prove observed
event counts or per-event identity. B counts; C proves identities/bytes.

Use psana2-gpu, psana-perlmutter and the existing implement-review-loop workflow.
New scope: up to three Claude review rounds, immutable commit and detached
review checkout, tests and scope supplied. Freeze reviewed script sources and
compare them to Git blobs before submitting. One lcls_g premium GPU-node job,
six-hour wall limit, preflight then main. Check personal/project balances.
Preserve exact settings, order, counts/unknowns, timing, cache, NIC, source and
scheduler evidence. Stop after submission is verified; report pending/running
accurately and collect results on follow-up.
