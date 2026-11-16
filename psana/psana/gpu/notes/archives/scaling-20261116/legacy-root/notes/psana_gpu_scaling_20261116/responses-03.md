# Compute-preflight correction for final review round

Previously approved code 92c07ce0570d7b1a8dfda8c4f0421fd347907833 was submitted as
job 61384230, which failed in 15 seconds on nid001068. All four CPU-probe steps
completed. The resource counts were correct (60/60/56/54 physical cores), but
actual step masks overlapped in only 53 physical cores. No native GPU tests,
pixel witnesses, cache-cleared timed cases, or throughput measurements ran.
Raw probes, failure log/status and terminal Slurm accounting are committed in
evidence/. The Lua module-hook warnings did not stop runtime activation and
are not the terminating failure; campaign reached the CPU mapping gate.

Correction: campaign_affinity requires a common 42-core prefix sufficient for
all ranks shared by up to three GPUs. Each profile puts those common cores
first; only the four additional BDs in the four-GPU case may consume that step's
extra cores. Prefix equality is explicitly checked for every shared rank.
Every profile still has three exclusive physical cores with both SMT siblings
per BD, and each helper retains three single-thread cores. The four-GPU step
must still fit 54 physical cores. Legacy one-GPU selection is unchanged.
Original probe files are now stored without mutation and compared byte-for-data
with the audit's records; the audit recomputes all maps with the same policy.

Regression: test_real_perlmutter_masks_preserve_shared_ranks uses all original
48 rank probes, confirms the original gate rejects their 53-core intersection,
then proves the corrected mapping fits each real step mask, uses exactly
18/30/42/54 distinct physical cores, and preserves all shared-rank placements.
A common set below 42 still fails. Full synthetic pilot/screen audits and report
checks remain unmocked except external input/hardware reads. All 17 tests pass
in 9.587 seconds; syntax checks pass. No framework or data changes.

Request final review of the updated exact commit, including regressions. User
session approval covers scoped fixes/reviews and job submission after checks.
No unreviewed code will be submitted. This is the third and final default round.
