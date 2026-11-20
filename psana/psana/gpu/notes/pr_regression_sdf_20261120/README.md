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

Validation: 145 offline tests passed after review corrections (validation-02.json). Job 46102038 completed all four small
CPU/GPU launch cases and a private 32 MiB cache/NIC control on sdfampere019 in
24 seconds. Both cache observations were zero. See validation-01.json for exact
NIC counts, source hashes, logs and earlier retained failed attempts. Cleanup
fault job 46102307 also passed all five cases in 2m11s. All 124 indexed artifacts
were independently rehashed; see cleanup-final.json. Intentional target-step
cancellations are expected fault-test outcomes; every sentinel completed 0:0.

This is implementation and bounded launch/network evidence, not a completed
run51 PR campaign. Actual PR builds still need source/API qualification, private
SDF input receipts, native smoke/accounting and correctness -> A/A -> A/B.
Observer overhead remains inconclusive. Prior Perlmutter studies are preserved.

Entry points: [generic handoff](../../docs/performance/templates/pr-performance-handoff.md)
and [SDF execution guide](../../scripts/performance/regression/SDF.md).

The maintained SDF `prepare_runtime.py capture`, `prepare_site.py`/`site_probe.py`,
`run_smoke.py`/`accounting_probe.py` and `native_worker.py` paths have not yet
executed on SDF hardware. Native launch evidence used a notes-only probe sharing
`launch()` and `sdf_rank.py`; the maintained cleanup driver did run successfully.
The first SDF campaign must qualify these remaining entry points before accepting
performance results. Small launch/NIC checks do not replace those qualifications.

Independent Claude review passed round 2 on `88f207a51837d13fc6395178137aa96ee9f353f1`; all round-1
findings are closed. See review-02.json and response-02.md. The subsequent commit
records evidence only. The first SDF campaign should supplement cleanup receipts
with post-cleanup PID liveness or scheduler proctrack/task-plugin evidence.
