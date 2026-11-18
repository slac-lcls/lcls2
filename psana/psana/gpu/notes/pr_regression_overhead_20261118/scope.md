# Instrumentation overhead qualification

Compare full runner accounting with a count-only observer, using the same pinned
runtime, private run-51 prefix, six raw cases and native launch/cache machinery.
This is not a PR A/B run or the subsequent clean A/A gate. No psana rebuild, MPI
cleanup change, calibrated workload or qualification-flag promotion is included.

Minimal retains event and GPU callback counts, boundary timers, synchronization,
provenance, placement, cache and NIC checks. It omits per-event identities/bytes,
step records, delivery wrappers, reader snapshots and callback-size lists. It
therefore cannot prove per-event identity or exact bytes for that sample. The
full side must pass those checks; both use independently verified immutable
inputs and the identical correctness-qualified runtime. Native psana counters
and the irreducible baseline counting/boundary costs are common and unmeasured.
No profiler, periodic stack dumping, payload D2H, or user kernel is enabled.

Fixed design: six pairs per case, alternating minimal/full then full/minimal,
10,000 measured events and 200 separate warmup events per sample. Analyze loop
wall time by the existing paired percentage-change Student-t 95% interval. Each
case must fit wholly within +/-1%; a negative out-of-band result also requires
investigation. All six must pass, with no missing/extra/excluded samples. No
optional stopping or automatic retries of measured samples. Cache conditioning
retains its documented bounded attempts before timing.

First run a 1,000-event, two-pair pilot of all six cases to check mechanics; it
cannot qualify overhead regardless of its interval. Final six-pair execution
requires the same reviewed source and pilot success. Any source change needs
review before measurement. Up to three independent Claude review rounds.

Preserve source/input hashes, all samples and failure evidence, scheduler exits,
cache attempts, CXI RX/TX, rank placement and timing. Distinguish mechanical
completion from statistical equivalence. Keep the MPI cleanup fix deferred.
