# Fixed-profile stripe sweep

Qualification61359629 and its independent results review passed. This directory
is a separate screen harness; accepted qualification code/artifacts remain
unchanged. The prior sandbox-helper failure is resolved by restarting the managed
Codex server; ordinary workspace and TMPDIR write/read/remove probes now pass.

Deferred-review changes:
- EXEC-R1-02:17tests now cover split200/1003tail callbacks, empty full-case BDs,
  missing/duplicate/wrong budgets, missing memory samples, duplicate Slurm limits.
- EXEC-R1-03: campaign and audit recompute the complete plan from the hashed ledger
  and sealed sizing evidence; edited values fail.
- EXEC-R1-04: deadline comes from UTC Slurm StartTime/EndTime with90s reserve.
- EXEC-R1-05: max(600,ceil(6*113.8344))=684s per case, twice measured preflight
  plus300s overhead reserve;18cases yield13020s=3h37m. This is a conservative
  failure/cost bound, not a measured bound for all layouts. Only count2 has
  full100k sizing evidence; slower layouts or cache failures can stop the sweep.
- EXEC-R1-06: sizing-evidence.json pins the accepted audit, complete receipt,
  all three timings and UTC accounting. Each raw hash matches the original
  committed artifact manifest and exact files in accepted results commit543240a9.
  Timings must also agree within10s with corresponding Slurm steps. Existing
  audit is preserved; new complete/audit receipts explicitly bind timing hashes.
- EXEC-R2-01: new audit records reviewed commit and complete.json hash.

Runtime/data path is unchanged; no rebuild. Files in evidence/ include copies
of the accepted sizing sources and verification record for independent review.
Attempt:/pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111/screen01.
No screen job submitted yet.
