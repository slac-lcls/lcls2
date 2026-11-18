# Full-prefix cache and traffic accounting qualification

Authorized next step after smoke 45e784267: investigate traffic-counter coverage
and the MPI exit warning, then run the six fixed raw cells with 10,000 events and
200-event separate warmups on the accepted private run-51 copy. Retain the known
runtime and all profile settings. No data copy, public-input eviction, package
edit, PR A/B, timing gate, overhead claim or qualification-flag promotion.

Implementation: extend the existing smoke driver with an explicit full-prefix
mode, keep default 1k behavior, rederive and hash the selected full prefix on
compute and compare its hashes with accepted preparation. Add read-only CXI HNI
RX/TX octet snapshots alongside the existing kernel netdev counters. Preserve
physical PCI mappings, timestamps, raw samples, missing/reset failures. Do not
sum HNI totals with subsets or with netdev totals.

Before the full cases, on the same allocation: two cold/warm controls on a private
256 MiB selected range, each cold pass evicted and checked twice with mincore;
warm pass must be fully resident. Compare data digests, preserve idle traffic.
Predeclared diagnostic coverage bounds: cold RX 0.9–2.0 times requested bytes;
warm and idle RX each at most 0.1 times requested bytes. These establish empirical
coverage for this allocation and read path, not exact storage attribution or a
link ceiling. Per full case require hardware RX at least 0.9 times selected bytes;
retain amplification rather than rejecting CPU read-ahead as datagram corruption.

MPI diagnosis: isolated three-rank controls (MPI-only; psana 20-event iteration
with Smd0.start return tracing; same with diagnostic Cancel/Wait of its specific
unmatched termination receive). Retain request and buffer while inspecting;
record pending state and cancellation outcome. Diagnostic code changes behavior
only in its separate process; full cases use the unchanged installed psana. A
warning disappearing is evidence for this path, not full lifetime qualification.

Checks: focused negative tests for counter parsing/reset/identity/missing cases
and full-prefix integrity; full existing offline suite; independent restricted
Claude review (maximum three rounds), bounded Perlmutter premium allocation with
account check, source/input/runtime pins, all steps COMPLETED 0:0, indexed raw
evidence, private cache eviction before each measured case, exact 10k event/byte
reference and non-idle BDs. Preserve failures. Final evidence review before push.
