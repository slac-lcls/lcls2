# SDF execution support and default cluster routing

The shared PR regression runner now selects SDF or Perlmutter execution from the
confirmed target cluster. The handoff defaults to the detected task cluster and
honors explicit remote targets. SDF uses one tracked Slurm step containing local
Open MPI 5 ranks, frozen per-rank CPU masks and one physical GPU UUID. SDF GPU
steps reserve the allocation's GPU set because exclusive allocations may grant
four devices even when one was requested; only the selected GPU is exposed to BDs.

Physical ethtool RX/TX byte counters are required on SDF. The runner preserves
raw snapshots, device identities, deltas, intervals and rates, rejects missing or
reset counters and reports per-sample network values alongside timing results.
Dataset roots are site-specific; private-copy/cache and event/byte verification
remain shared. No qualification flags or performance tolerances were relaxed.

Validation: 144 offline tests passed. Job 46102038 completed all four small
CPU/GPU launch cases and a private 32 MiB cache/NIC control on sdfampere019 in
24 seconds. Both cache observations were zero. See validation-01.json for exact
NIC counts, source hashes, logs and earlier retained failed attempts. Cleanup
fault checks are tracked as job 46102307; final status will be recorded separately.

This is implementation and bounded launch/network evidence, not a completed
run51 PR campaign. Actual PR builds still need source/API qualification, private
SDF input receipts, native smoke/accounting and correctness -> A/A -> A/B.
Observer overhead remains inconclusive. Prior Perlmutter studies are preserved.

Entry points: [generic handoff](../../docs/performance/templates/pr-performance-handoff.md)
and [SDF execution guide](../../scripts/performance/regression/SDF.md).
